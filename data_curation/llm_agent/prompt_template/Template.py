import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(cur_dir, "QueryRewriterTemplate.txt")) as f:
    QueryRewriterTemplate = "".join(f.readlines())

with open(os.path.join(cur_dir, "QueryJudgerTemplate.txt")) as f:
    QueryJudgerTemplate = "".join(f.readlines())

with open(os.path.join(cur_dir, "QueryGeneratorTemplate.txt")) as f:
    QueryGeneratorTemplate = "".join(f.readlines())

with open(os.path.join(cur_dir, "UnambiguousGeneratorTemplateLong.txt")) as f:
    UnambiguousGeneratorTemplateLong = "".join(f.readlines())

with open(os.path.join(cur_dir, "UnambiguousGeneratorTemplateShort.txt")) as f:
    UnambiguousGeneratorTemplateShort = "".join(f.readlines())

with open(os.path.join(cur_dir, "DecomposeGeneratorTemplate.txt")) as f:
    DecomposeGeneratorTemplate = "".join(f.readlines())

with open(os.path.join(cur_dir, "DecomposeGeneratorTemplateForAns.txt")) as f:
    DecomposeGeneratorTemplateForAns = "".join(f.readlines())

with open(os.path.join(cur_dir, "MultiTurnGeneratorTemplate.txt")) as f:
    MultiTurnGeneratorTemplate = "".join(f.readlines())

with open(os.path.join(cur_dir, "MultiTurnGeneratorTemplateForAns.txt")) as f:
    MultiTurnGeneratorTemplateForAns = "".join(f.readlines())

with open(os.path.join(cur_dir, "QAGeneratorTemplate.txt")) as f:
    QAGeneratorTemplate = "".join(f.readlines())

if __name__ == '__main__':

    print(QueryRewriterTemplate)
    print(QueryJudgerTemplate)
    print(QueryGeneratorTemplate)
