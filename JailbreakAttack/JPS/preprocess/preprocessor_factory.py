from .internvl2_8b_preprocessor import Internvl28BPreprocessor
from .internvl2_5_8b_preprocessor import Internvl2_58BPreprocessor
from .internvl3_8b_preprocessor import Internvl38BPreprocessor
from .internvl3_5_8b_preprocessor import Internvl3_58BPreprocessor
from .internvl3_5_38b_preprocessor import Internvl3_5_38BPreprocessor
from .qwen2vl_7b_preprocessor import Qwen2VL7BPreprocessor
from .qwen2_5vl_7b_preprocessor import Qwen2_5VL7BPreprocessor
from .qwen3vl_8b_preprocessor import Qwen3VL8BPreprocessor
from .qwen3vl_32b_preprocessor import Qwen3VL32BPreprocessor

class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(config):
        model_name = config['model']['name']
        if model_name  == 'internvl2_8b':
            return Internvl28BPreprocessor(config['preprocessing'])
        if model_name == 'internvl2_5_8b':
            return Internvl2_58BPreprocessor(config['preprocessing'])
        if model_name == 'internvl3_8b':
            return Internvl38BPreprocessor(config['preprocessing'])
        if model_name == 'internvl3_5_8b':
            return Internvl3_58BPreprocessor(config['preprocessing'])
        if model_name == 'internvl3_5_38b':
            return Internvl3_5_38BPreprocessor(config['preprocessing'])
        if model_name == 'qwen2vl_7b':
            return Qwen2VL7BPreprocessor(config['preprocessing'])
        if model_name == 'qwen2_5vl_7b':
            return Qwen2_5VL7BPreprocessor(config['preprocessing'])
        if model_name == 'qwen3vl_8b':
            return Qwen3VL8BPreprocessor(config['preprocessing'])
        if model_name == 'qwen3vl_32b':
            return Qwen3VL32BPreprocessor(config['preprocessing'])
        else:
            raise ValueError(f"未知的模型名称: {model_name}")