from collections import OrderedDict

from transformers import BertConfig, AlbertConfig, ElectraConfig, RobertaConfig
from transformers import XLMConfig, DistilBertConfig, CamembertConfig, XLMRobertaConfig
from transformers import AutoConfig, PretrainedConfig

from models.albert_ner import AlbertCrfForNer, AlbertSoftmaxForNer, AlbertSpanForNer
from models.bert_ner import BertCrfForNer, BertSoftmaxForNer, BertSpanForNer
from models.electra_ner import ElectraCrfForNer, ElectraSoftmaxForNer, ElectraSpanForNer
from models.roberta_ner import RobertaCrfForNer, RobertaSoftmaxForNer, RobertaSpanForNer
from models.xlm_ner import XLMCrfForNer, XLMSoftmaxForNer, XLMSpanForNer
from models.distilbert_ner import DistilBertCrfForNer, DistilBertSoftmaxForNer, DistilBertSpanForNer

MODEL_FOR_SOFTMAX_NER_MAPPING = OrderedDict(
    [
        (XLMConfig, XLMSoftmaxForNer),
        (DistilBertConfig, DistilBertSoftmaxForNer),
        (RobertaConfig, RobertaSoftmaxForNer),
        (CamembertConfig, RobertaSoftmaxForNer),
        (XLMRobertaConfig, RobertaSoftmaxForNer),
        (BertConfig, BertSoftmaxForNer),
        (AlbertConfig, AlbertSoftmaxForNer),
        (ElectraConfig, ElectraSoftmaxForNer),
    ]
)

MODEL_FOR_CRF_NER_MAPPING = OrderedDict(
    [
        (XLMConfig, XLMCrfForNer),
        (DistilBertConfig, DistilBertCrfForNer),
        (RobertaConfig, RobertaCrfForNer),
        (CamembertConfig, RobertaCrfForNer),
        (XLMRobertaConfig, RobertaCrfForNer),
        (BertConfig, BertCrfForNer),
        (AlbertConfig, AlbertCrfForNer),
        (ElectraConfig, ElectraCrfForNer),
    ]
)

MODEL_FOR_SPAN_NER_MAPPING = OrderedDict(
    [
        (XLMConfig, XLMSpanForNer),
        (DistilBertConfig, DistilBertSpanForNer),
        (RobertaConfig, RobertaSpanForNer),
        (CamembertConfig, RobertaSpanForNer),
        (XLMRobertaConfig, RobertaSpanForNer),
        (BertConfig, BertSpanForNer),
        (AlbertConfig, AlbertSpanForNer),
        (ElectraConfig, ElectraSpanForNer),
    ]
)


class AutoModelForSoftmaxNer:
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTokenClassification is designed to be instantiated "
            "using the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_SOFTMAX_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SOFTMAX_NER_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_SOFTMAX_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SOFTMAX_NER_MAPPING.keys()),
            )
        )


class AutoModelForCrfNer:
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTokenClassification is designed to be instantiated "
            "using the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_CRF_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_CRF_NER_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_CRF_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_CRF_NER_MAPPING.keys()),
            )
        )


class AutoModelForSpanNer:
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTokenClassification is designed to be instantiated "
            "using the `AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForTokenClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_SPAN_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SPAN_NER_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_SPAN_NER_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SPAN_NER_MAPPING.keys()),
            )
        )
