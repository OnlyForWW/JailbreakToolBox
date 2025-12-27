from .llamaguard3_judge import LlamaGuard3Judge

class JudgeFactory:
    @staticmethod
    def get_judger(config, model_name):
        if model_name  == 'llamaguard3':
            return LlamaGuard3Judge(config)
        else:
            raise ValueError(f"未知的模型名称: {model_name}")