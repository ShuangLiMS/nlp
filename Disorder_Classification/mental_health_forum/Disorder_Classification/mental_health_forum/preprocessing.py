import regex as re
import jamspell
import logging_util

logger = logging_util.logger(__name__)

def rm_punctuation(paragraph):
    """
    Remove punctuation
    :param paragraph:
    :return:
    """
    return " ".join(re.split(r"\s+|[!,;:?.'-]\s*", paragraph))

def spell_corr():
    corrector = jamspell.TSpellCorrector()
    corrector.LoadLangModel('model_en.bin')
    logger.info(f"corrector created: {result}")
    return corrector

