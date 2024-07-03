from pathlib import Path
from packaging import version

from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from src.util.param import parse_args
from pretrain_data import get_loader
from src.util.utils import LossMeter
from src.util.dist_utils import reduce_dict

from torch.utils.tensorboard import SummaryWriter

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase


# The Trainer inherits TrainerBase in trainer_base.py
class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        assert args.whole_word_embed
        from pretrain_model import Q5Pretraining, RMPretraining

        model_kwargs = {}
        # model_class = P5Pretraining
        # model_class = Q5Pretraining
        model_class = RMPretraining

        print('init trainer')

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        # 这里没有涉及snap中的路径，到底是怎么加载的呢
        # 调用了arg里面的路径
        self.model = self.create_model(model_class, config, **model_kwargs)

        if 'p5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)

        self.model.tokenizer = self.tokenizer

        self.writer = SummaryWriter('./tf_logs/')

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)
            self.start_epoch = int(args.load.split('Epoch-')[-1])

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.args.dry:
            results = self.evaluate_epoch(epoch=0)

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 100000.

            if 't5' in self.args.backbone:
                project_name = "P5_Pretrain"

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        for epoch in range(self.args.epoch):
            if self.start_epoch is not None:
                epoch += self.start_epoch
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            self.model.train()

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=275)

            epoch_results = {}
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # 写入文件
                self.writer.add_scalar('mAP', loss.detach().cpu().data, global_step)
                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                if self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # self.model.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                global_step += 1

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 200:
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.verbose:
                pbar.close()

            dist.barrier()

            results = reduce_dict(epoch_results, average=False)
            if self.verbose:
                train_loss = results['total_loss']
                train_loss_count = results['total_loss_count']

                avg_train_loss = train_loss / train_loss_count
                losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                for name, loss in results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(results[name + '_count'])
                        if loss_count > 0:
                            avg_loss = loss / loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                losses_str += '\n'
                print(losses_str)

            dist.barrier()

            if epoch > 0:
                # Validation
                valid_results = self.evaluate_epoch(epoch=epoch)

                valid_results = reduce_dict(valid_results, average=False)
                if self.verbose and step_i % 200:
                    valid_loss = valid_results['total_loss']
                    valid_loss_count = valid_results['total_loss_count']

                    avg_valid_loss = valid_loss / valid_loss_count
                    losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                    for name, loss in valid_results.items():
                        if name[-4:] == 'loss':
                            loss_count = int(valid_results[name + '_count'])
                            if loss_count > 0:
                                avg_loss = loss / loss_count
                                losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                    losses_str += '\n'
                    print(losses_str)

                dist.barrier()

                if self.verbose:
                    # Save
                    if avg_valid_loss < best_eval_loss:
                        best_eval_loss = avg_valid_loss
                        self.save("BEST_EVAL_LOSS")
                    self.save("Epoch%02d" % (epoch + 1))

                dist.barrier()
            else:
                # Skip validation
                print("Skip validation for Epoch%02d" % (epoch + 1))
                # 比较低的epoch就不用save了
                self.save("Epoch%02d" % (epoch + 1))

                dist.barrier()

    def evaluate_epoch(self, epoch):
        LOSSES_NAME = self.args.LOSSES_NAME

        epoch_results = {}
        for loss_name in LOSSES_NAME:
            epoch_results[loss_name] = 0.
            epoch_results[f'{loss_name}_count'] = 0

        self.model.eval()
        with torch.no_grad():
            if self.verbose:
                loss_meter = LossMeter()
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                pbar = tqdm(total=len(self.val_loader), ncols=275)

            for step_i, batch in enumerate(self.val_loader):
                # print(batch)

                if self.args.distributed:
                    results = self.model.module.valid_step(batch)
                else:
                    results = self.model.valid_step(batch)

                for k, v in results.items():
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 200:
                    desc_str = f'Valid Epoch {epoch} |'
                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results:
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                dist.barrier()

            if self.verbose:
                pbar.close()
            dist.barrier()

            return epoch_results


def main_worker(gpu, args):
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    datasplitmap = {'text': 1, 'QAC': 0.05, 'Q2Q': 1, 'I2Q': 1, 'Q2C': 0.4, 'U2C': 0.05, 'U2QC': 1, 'traditional': 1,
                    'Q2T': 1, 'Q2TR': 1, 'Q2TC': 1, 'RM': 1, }
    # datasplitmap = {'text': 0.5, 'QAC': 0.025, 'Q2Q': 0.5, 'I2Q': 0.5, 'Q2C': 0.2, 'U2C': 0.025, 'U2QC': 0.5,
    #                 'traditional': 1, 'Q2TR': 0.5, 'Q2TC': 0.5, }

    print(f'Building train loader at GPU {gpu}')
    # define the prompts used in training
    if args.train == 'test':
        train_task_list = {
            'Q2Q': ['5-9', '5-10', '5-11'],
            'I2Q': ['6-4', '6-5', '6-6'],
        }
    elif args.train == 'icbu':
        train_task_list = {
            'sequential': ['2-1', '2-2', '2-3'],
        }
    elif args.train == 'icbu_text':
        datasplitmap['text'] = 1
        train_task_list = {
            'text': ['3-7', '3-8', '3-9'],
        }
    elif args.train == 'icbu_texttra':
        train_task_list = {
            'text': ['3-10', '3-11', '3-12', '3-13', '3-14'],
            'traditional': ['10-1', '10-2'],
        }
    elif args.train == 'QAC':
        datasplitmap['QAC'] = 1
        train_task_list = {
            'QAC': ['4-4', '4-5', '4-6'],
        }
    elif args.train == 'Q2Q':
        datasplitmap['Q2Q'] = 1
        train_task_list = {
            'Q2Q': ['5-5', '5-6', '5-7', '5-8'],
        }
    elif args.train == 'I2Q':
        datasplitmap['I2Q'] = 1
        train_task_list = {
            'I2Q': ['6-1', '6-2', '6-3'],
        }
    elif args.train == 'Q2C':
        datasplitmap['Q2C'] = 1
        train_task_list = {
            'Q2C': ['8-3', '8-4'],
        }
    elif args.train == 'U2C':
        datasplitmap['U2C'] = 1
        train_task_list = {
            'U2C': ['9-3', '9-4'],
        }
    elif args.train == 'U2QC':
        datasplitmap['U2QC'] = 1
        train_task_list = {
            'U2QC': ['7-5'],
        }
    elif args.train == 'Q2T':
        datasplitmap['Q2T'] = 1
        train_task_list = {
            'Q2T': ['11-4'],
        }
    elif args.train == 'Q2TR':
        datasplitmap['Q2T'] = 1
        train_task_list = {
            'Q2TR': ['11-1', '11-2'],
        }
    elif args.train == 'Q2TC':
        datasplitmap['Q2T'] = 1
        train_task_list = {
            'Q2TC': ['11-4', ],
        }
    elif args.train == 'RM':
        datasplitmap['RM'] = 1
        train_task_list = {
            'RM': ['12-3', ],
        }
    elif args.train == 'icbu_seqtext':
        train_task_list = {
            'sequential': ['2-1', '2-2', '2-3'],
            'text': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6'],
        }
    elif args.train == 'icbu_seqtextqac':
        train_task_list = {
            'sequential': ['2-1', '2-2', '2-3'],
            'text': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6'],
            'QAC': ['4-4', '4-5', '4-6'],
        }
    elif args.train == 'icbu_seqtextqacq2q':
        train_task_list = {
            'sequential': ['2-1', '2-2', '2-3'],
            'text': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6'],
            'QAC': ['4-4', '4-5', '4-6'],
            'Q2Q': ['5-1', '5-2', '5-3', '5-4'],
        }
    elif args.train == 'icbu_textqacq2qi2q':
        train_task_list = {
            'text': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6'],
            'QAC': ['4-4', '4-5', '4-6'],
            'Q2Q': ['5-1', '5-2', '5-3', '5-4'],
            'I2Q': ['6-1', '6-2', '6-3'],
        }
    elif args.train == 'icbu_u2qqacq2qi2q':
        train_task_list = {
            'text': ['3-7', '3-8', '3-9'],
            'QAC': ['4-4', '4-5', '4-6'],
            'Q2Q': ['5-5', '5-6', '5-7', '5-8'],
            'I2Q': ['6-1', '6-2', '6-3'],
        }
    elif args.train == 'icbu_u2qqacq2qi2qq2c':
        train_task_list = {
            'text': ['3-10', '3-11', '3-12'],
            'QAC': ['4-7', '4-8', '4-9'],
            'Q2Q': ['5-9', '5-10', '5-11'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-3', '8-4'],
        }
    elif args.train == 'icbu_u2qqacq2qi2qq2ctra':
        train_task_list = {
            'text': ['3-10', '3-11', '3-12', '3-13', '3-14', '3-15', '3-16'],
            'QAC': ['4-7', '4-8', '4-9', '4-10', '4-11'],
            'Q2Q': ['5-9', '5-10', '5-11', '5-12', '5-13'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-3', '8-4', ],
            'traditional': ['10-1', '10-2', ],
        }
    elif args.train == 'icbu_u2qqacq2qi2qq2cu2c':
        train_task_list = {
            'text': ['3-10', '3-11', '3-12'],
            'QAC': ['4-7', '4-8', '4-9'],
            'Q2Q': ['5-9', '5-10', '5-11'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-3', '8-4'],
            'U2C': ['9-1', '9-2'],
        }
    elif args.train == 'icbu_u2qqacq2qi2qq2cu2c-v2':
        train_task_list = {
            'text': ['3-10', '3-11', '3-12', '3-13', '3-14'],
            'QAC': ['4-7', '4-8', '4-9', '4-10'],
            'Q2Q': ['5-9', '5-10', '5-11', '5-12'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-3', '8-4', '8-5'],
            'U2C': ['9-1', '9-2', '9-3', '9-4'],
        }
    elif args.train == 'icbu_u2qqacq2qi2qq2cu2ctra':
        train_task_list = {
            'text': ['3-10', '3-11', '3-12', '3-13', '3-14'],
            'QAC': ['4-7', '4-8', '4-9', '4-10'],
            'Q2Q': ['5-9', '5-10', '5-11', '5-12'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-6'],
            'U2C': ['9-3', '9-4'],
            'traditional': ['10-1', '10-2', '10-3', '10-4'],
        }
    elif args.train == 'icbu_u2qqacq2qi2qq2cu2cq2ttra':
        train_task_list = {
            'text': ['3-10', '3-11', '3-12', '3-13', '3-14'],
            'QAC': ['4-7', '4-8', '4-9', '4-10'],
            'Q2Q': ['5-9', '5-10', '5-11', '5-12'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-3', '8-4'],
            'U2C': ['9-3', '9-4'],
            'Q2T': ['11-1', '11-2'],
            'traditional': ['10-1', '10-2', '10-3', '10-4'],
        }
    elif args.train == 'icbu_u2qqacq2qi2qq2cu2cq2trq2tctra':
        train_task_list = {
            'text': ['3-10', '3-11', '3-12', '3-13', '3-14'],
            'QAC': ['4-7', '4-8', '4-9', '4-10'],
            'Q2Q': ['5-9', '5-10', '5-11', '5-12'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-3', '8-4'],
            'U2C': ['9-3', '9-4'],
            'Q2TR': ['11-1', '11-2'],
            'Q2TC': ['11-4', ],
            'traditional': ['10-1', '10-2', '10-3', '10-4'],
        }
    else:
        raise NotImplementedError

    # define sampling numbers for each group of personalized prompts (see pretrain_data.py)
    # if greater than 1, a data sample will be used for multiple times with different prompts in certain task family
    train_sample_numbers = {'rating': 1, 'sequential': (5, 5, 10), 'explanation': 1, 'review': 1,
                            'traditional': (10, 5)}
    train_loader = get_loader(
        args,
        train_task_list,
        train_sample_numbers,
        split=args.train,
        mode='train',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed,
        datasplitmap=datasplitmap,
    )

    print(f'Building val loader at GPU {gpu}')
    # define the prompts used in validation
    if args.valid == 'test':
        val_task_list = {
            'Q2Q': ['5-9', '5-10', '5-11'],
            'I2Q': ['6-4', '6-5', '6-6'],
        }
    elif args.valid == 'icbu':
        val_task_list = {
            'sequential': ['2-1', '2-2', '2-3'],
        }
    elif args.valid == 'icbu_text':
        val_task_list = {
            'text': ['3-7', '3-8', '3-9'],
        }
    elif args.valid == 'icbu_texttra':
        val_task_list = {
            'text': ['3-10', '3-11', '3-12', '3-13', '3-14'],
            'traditional': ['10-1', '10-2', '10-3', '10-4'],
        }
    elif args.valid == 'QAC':
        val_task_list = {
            'QAC': ['4-4', '4-5', '4-6'],
        }
    elif args.valid == 'Q2Q':
        val_task_list = {
            'Q2Q': ['5-5', '5-6', '5-7', '5-8'],
        }
    elif args.valid == 'I2Q':
        val_task_list = {
            'I2Q': ['6-1', '6-2', '6-3'],
        }
    elif args.valid == 'Q2C':
        val_task_list = {
            'Q2C': ['8-3', '8-4'],
        }
    elif args.valid == 'U2C':
        val_task_list = {
            'U2C': ['9-3', '9-4'],
        }
    elif args.valid == 'U2QC':
        val_task_list = {
            'U2QC': ['7-5', ],
        }
    elif args.valid == 'Q2T':
        val_task_list = {
            'Q2T': ['11-4', ],
        }
    elif args.valid == 'RM':
        val_task_list = {
            'RM': ['12-3', ],
        }
    elif args.valid == 'icbu_seqtext':
        val_task_list = {
            'sequential': ['2-1', '2-2', '2-3'],
            'text': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6'],
        }
    elif args.valid == 'icbu_seqtextqac':
        val_task_list = {
            'sequential': ['2-1', '2-2', '2-3'],
            'text': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6'],
            'QAC': ['4-4', '4-5', '4-6'],
        }
    elif args.valid == 'icbu_seqtextqacq2q':
        val_task_list = {
            'sequential': ['2-1', '2-2', '2-3'],
            'text': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6'],
            'QAC': ['4-4', '4-5', '4-6'],
            'Q2Q': ['5-1', '5-2', '5-3', '5-4'],
        }
    elif args.valid == 'icbu_textqacq2qi2q':
        val_task_list = {
            'text': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6'],
            'QAC': ['4-4', '4-5', '4-6'],
            'Q2Q': ['5-1', '5-2', '5-3', '5-4'],
            'I2Q': ['6-1', '6-2', '6-3'],
        }
    elif args.valid == 'icbu_u2qqacq2qi2q':
        val_task_list = {
            'text': ['3-7', '3-8', '3-9'],
            'QAC': ['4-4', '4-5', '4-6'],
            'Q2Q': ['5-5', '5-6', '5-7', '5-8'],
            'I2Q': ['6-1', '6-2', '6-3'],
        }
    elif args.valid == 'icbu_u2qqacq2qi2qq2c':
        val_task_list = {
            'text': ['3-10', '3-11', '3-12'],
            'QAC': ['4-7', '4-8', '4-9'],
            'Q2Q': ['5-9', '5-10', '5-11'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-3', '8-4'],
        }
    elif args.valid == 'icbu_u2qqacq2qi2qq2ctra':
        val_task_list = {
            'text': ['3-10', '3-11', '3-12', '3-13', '3-14'],
            'QAC': ['4-7', '4-8', '4-9', '4-10'],
            'Q2Q': ['5-9', '5-10', '5-11', '5-12'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-3', '8-4', '8-5'],
            'traditional': ['10-1', '10-2', '10-3', '10-4'],
        }
    elif args.valid == 'icbu_u2qqacq2qi2qq2cu2c':
        val_task_list = {
            'text': ['3-10', '3-11', '3-12'],
            'QAC': ['4-7', '4-8', '4-9'],
            'Q2Q': ['5-9', '5-10', '5-11'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-3', '8-4'],
            'U2C': ['9-1', '9-2'],
        }
    elif args.valid == 'icbu_u2qqacq2qi2qq2cu2c-v2':
        val_task_list = {
            'text': ['3-10', '3-11', '3-12', '3-13', '3-14'],
            'QAC': ['4-7', '4-8', '4-9', '4-10'],
            'Q2Q': ['5-9', '5-10', '5-11', '5-12'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-3', '8-4', '8-5'],
            'U2C': ['9-1', '9-2', '9-3', '9-4'],
        }
    elif args.valid == 'icbu_u2qqacq2qi2qq2cu2ctra':
        val_task_list = {
            'text': ['3-10', '3-11', '3-12', '3-13', '3-14'],
            'QAC': ['4-7', '4-8', '4-9', '4-10'],
            'Q2Q': ['5-9', '5-10', '5-11', '5-12'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-6'],
            'U2C': ['9-3', '9-4'],
            'traditional': ['10-1', '10-2', '10-3', '10-4'],
        }
    elif args.valid == 'icbu_u2qqacq2qi2qq2cu2cq2ttra':
        val_task_list = {
            'text': ['3-10', '3-11', '3-12', '3-13', '3-14'],
            'QAC': ['4-7', '4-8', '4-9', '4-10'],
            'Q2Q': ['5-9', '5-10', '5-11', '5-12'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-3', '8-4'],
            'U2C': ['9-3', '9-4'],
            'Q2T': ['11-1', '11-2'],
            'traditional': ['10-1', '10-2', '10-3', '10-4'],
        }
    elif args.valid == 'icbu_u2qqacq2qi2qq2cu2cq2trq2tctra':
        val_task_list = {
            'text': ['3-10', '3-11', '3-12', '3-13', '3-14'],
            'QAC': ['4-7', '4-8', '4-9', '4-10'],
            'Q2Q': ['5-9', '5-10', '5-11', '5-12'],
            'I2Q': ['6-4', '6-5', '6-6'],
            'Q2C': ['8-3', '8-4'],
            'U2C': ['9-3', '9-4'],
            'Q2TR': ['11-1', '11-2'],
            'Q2TC': ['11-4', ],
            'traditional': ['10-1', '10-2', '10-3', '10-4'],
        }
    else:
        raise NotImplementedError

    val_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
    val_loader = get_loader(
        args,
        val_task_list,
        val_sample_numbers,
        split=args.valid,
        mode='val',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed,
        datasplitmap=datasplitmap,
    )

    trainer = Trainer(args, train_loader, val_loader, train=True)
    trainer.train()


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    if args.local_rank in [0, -1]:
        print(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]  # rating, seq, explain
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss')

    args.LOSSES_NAME = LOSSES_NAME

    comments = []
    dsets = []
    if 'u2q' in args.train:
        dsets.append('U2Q')
    if 'qac' in args.train:
        dsets.append('QAC')
    if 'q2q' in args.train:
        dsets.append('Q2Q')
    if 'i2q' in args.train:
        dsets.append('I2Q')
    if 'q2c' in args.train:
        dsets.append('Q2C')
    if 'u2c' in args.train:
        dsets.append('U2C')
    if 'u2qc' in args.train:
        dsets.append('U2QC')
    if 'q2t' in args.train:
        dsets.append('Q2T')
    if 'rm' in args.train:
        dsets.append('RM')
    if 'traditional' in args.train:
        dsets.append('traditional')

    comments.append(''.join(dsets))
    if args.backbone:
        comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    from datetime import datetime

    current_time = datetime.now().strftime('%b%d_%H-%M')

    project_dir = Path(__file__).resolve().parent.parent

    if args.local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)

