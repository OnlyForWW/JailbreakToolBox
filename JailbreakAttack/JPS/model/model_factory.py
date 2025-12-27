class ModelFactory:
    @staticmethod
    def get_model(config):
        model_name = config['model']['name']
        if model_name  == 'internvl2_8b':
            from .internvl2_8b_model import Internvl28BModel
            return Internvl28BModel(config)
        if model_name == 'internvl2_5_8b':
            from .internvl2_5_8b_model import Internvl2_58BModel
            return Internvl2_58BModel(config)
        if model_name == 'internvl3_8b':
            from .internvl3_8b_model import Internvl38BModel
            return Internvl38BModel(config)
        if model_name == 'internvl3_5_8b':
            from .internvl3_5_8b_model import Internvl3_58BModel
            return Internvl3_58BModel(config)
        if model_name == 'internvl3_5_38b':
            from .internvl3_5_38b_model import Internvl3_5_38BModel
            return Internvl3_5_38BModel(config)
        if model_name == 'qwen2vl_7b':
            from .qwen2vl_7b_model import Qwen2VL7BModel
            return Qwen2VL7BModel(config)
        if model_name == 'qwen2_5vl_7b':
            from .qwen2_5vl_7b_model import Qwen2_5VL7BModel
            return Qwen2_5VL7BModel(config)
        if model_name == 'qwen3vl_8b':
            from .qwen3vl_8b_model import Qwen3VL8BModel
            return Qwen3VL8BModel(config)
        if model_name == 'qwen3vl_32b':
            from .qwen3vl_32b_model import Qwen3VL32BModel
            return Qwen3VL32BModel(config)
        else:
            raise ValueError(f"未知的模型名称: {model_name}")