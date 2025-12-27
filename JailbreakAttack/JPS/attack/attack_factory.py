class AttackFactory:
    @staticmethod
    def get_attack(config, pipeline):
        model_name = config['model']['name']
        if model_name  == 'internvl2_8b':
            from .internvl2_8b_attack import Internvl28BAttack
            return  Internvl28BAttack(config, pipeline)
        if model_name == 'internvl2_5_8b':
            from .internvl2_5_8b_attack import Internvl2_58BAttack
            return Internvl2_58BAttack(config, pipeline)
        if model_name == 'internvl3_8b':
            from .internvl3_8b_attack import Internvl38BAttack
            return Internvl38BAttack(config, pipeline)
        if model_name == 'internvl3_5_8b':
            from .internvl3_5_8b_attack import Internvl3_58BAttack
            return Internvl3_58BAttack(config, pipeline)
        if model_name == 'internvl3_5_38b':
            from .internvl3_5_38b_attack import Internvl3_5_38BAttack
            return Internvl3_5_38BAttack(config, pipeline)
        if model_name == 'qwen2vl_7b':
            from .qwen2vl_7b_attack import Qwen2VL7BAttack
            return Qwen2VL7BAttack(config, pipeline)
        if model_name == 'qwen2_5vl_7b':
            from .qwen2_5vl_7b_attack import Qwen2_5VL7BAttack
            return Qwen2_5VL7BAttack(config, pipeline)
        if model_name == 'qwen3vl_8b':
            from .qwen3vl_8b_attack import Qwen3VL8BAttack
            return Qwen3VL8BAttack(config, pipeline)
        if model_name == 'qwen3vl_32b':
            from .qwen3vl_32b_attack import Qwen3VL32BAttack
            return Qwen3VL32BAttack(config, pipeline)
        else:
            raise ValueError(f"未知的模型名称: {model_name}")