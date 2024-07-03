from .U2Q import task_subgroup as task_subgroup3
from .QAC import task_subgroup as task_subgroup4
from .Q2Q import task_subgroup as task_subgroup5
from .I2Q import task_subgroup as task_subgroup6
from .U2QC import task_subgroup as task_subgroup7
from .Q2C import task_subgroup as task_subgroup8
from .U2C import task_subgroup as task_subgroup9
from .traditional import task_subgroup as task_subgroup10
from .Q2T import task_subgroup as task_subgroup11
from .eval import eval_tasks
from .RM import task_subgroup as task_subgroup12
from .PRO import task_subgroup as task_subgroup13

all_tasks = {
    'text': task_subgroup3,
    'QAC': task_subgroup4,
    'Q2Q': task_subgroup5,
    'I2Q': task_subgroup6,
    'Q2C': task_subgroup8,
    'U2C': task_subgroup9,
    'U2QC': task_subgroup7,
    'traditional': task_subgroup10,
    'Q2T': task_subgroup11,
    'RM': task_subgroup12,
    'PRO': task_subgroup13,
}
