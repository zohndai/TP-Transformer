import codecs
import glob
import gc
import torch
from collections import Counter, defaultdict
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _build_fields_vocab, _load_vocab, DatasetLazyIter, OrderedIterator
from functools import partial
from multiprocessing import Pool
from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab
import onmt.translate
import pandas as pd
import onmt.model_builder
import onmt.translate
from onmt.utils.misc import split_corpus

parser_tsl = ArgumentParser(description="translate.py")
opts.config_opts(parser_tsl)
opts.translate_opts(parser_tsl)
args_tsl = ['-model', 'D:/Desktop/data/finnal-datasets-for-paper/Chem_Oxi_2k/20240424_best_model/modify3/transfer/transfer_step_47700.pt', \
            '-src', 'D:/Desktop/data/finnal-datasets-for-paper/Chem_Oxi_2k/20240424_best_model/modify3/mini/fmy.txt', \
            '-output', 'predictions.txt', \
            '-n_best', '5', \
            '-beam_size', '10', \
            '-max_length', '3000', \
            '-batch_size', '64']
opt_tsl = parser_tsl.parse_args(args_tsl)
ArgumentParser.validate_translate_opts(opt_tsl)
logg1er = init_logger(opt_tsl.log_file)

def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])#加载模型参数
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):#False
        fields = inputters.load_old_vocab(vocab, opt.data_type, dynamic_dict=model_opt.copy_attn)
    else:
        fields = vocab
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint, opt.gpu)
    if opt.fp32:#False
        model.float()
    model.eval()
    model.generator.eval()
    return fields, model, model_opt

def build_translator(opt, report_score, logger=None, out_file=None):
    log_probs_out_file = None
    target_score_out_file = None
    if out_file is None:#True
        out_file = codecs.open(opt.output, 'w+', 'utf-8')
        
    if opt.log_probs:#False
        log_prob_oout_file = codecs.open(opt.ouput + '_log_probs', 'w+', 'utf-8')
        target_score_out_file = codecs.open(opt.output + '_gold_score', 'w+', 'utf-8')
        
    load_test_model = onmt.decoders.ensemble.load_test_model if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)
    
    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)
    
    translator = onmt.translate.Translator.from_opt(
        model, fields, opt, model_opt, global_scorer=scorer, out_file=out_file, report_align=opt.report_align, \
        report_score=report_score, logger=logger, log_probs_out_file=log_probs_out_file, target_score_out_file=target_score_out_file,
    )
    return translator

model_path = opt_tsl.models[0]
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
vocab = checkpoint['vocab']
translator = build_translator(opt_tsl, report_score=True)
src_shards_tsl = split_corpus(opt_tsl.src, opt_tsl.shard_size) # list(islice(f, shard_size))
tgt_shards_tsl = split_corpus(opt_tsl.tgt, opt_tsl.shard_size)
shard_pairs_tsl = zip(src_shards_tsl, tgt_shards_tsl)
for i, (src_shard_tsl, tgt_shard_tsl) in enumerate(shard_pairs_tsl): # 0, ([src], None)
    logger.info("Translating shard %d." % i) # 只有0
    translator.translate(
        src=src_shard_tsl,
        tgt=tgt_shard_tsl,
        src_dir=opt_tsl.src_dir, #''
        batch_size=opt_tsl.batch_size, # 64
        batch_type=opt_tsl.batch_type, # sents
        attn_debug=opt_tsl.attn_debug, # False
        align_debug=opt_tsl.align_debug # False
    )

dp_smis = pd.read_csv(opt_tsl.output,header=None)
smis_li=["".join(dp_smi.split(" ")) for dp_smi in dp_smis[0]]
print(smis_li)