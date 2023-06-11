# -*- coding: utf-8 -*-
import os
import time
from multiprocessing import Process, Lock

from nameko.standalone.rpc import ClusterRpcProxy

from service.kfka_consumer import insert_mysql

config_mq = {'AMQP_URI': "amqp://guest:guest@127.0.0.1"}

if __name__ == '__main__':
    url = 'https://new.qq.com/omn/20220321/20220321A04RQA00.html'
    src = url.split('/')[-1].replace('.html', '')
    print(src)

    # batches = [['20220322', 'https://new.qq.com/omn/20220322/20220322A0373V00.html', '辽博书画谁堪比',
    #         '2018年，辽宁省博物馆中国古代绘画展展出簪花仕女图。视觉中国供图2019年10月，名为又见大唐的书画文物展在辽宁省博物馆开幕，两月展期，各地观众纷至沓来，火爆一时。又见大唐，展品自然以唐代文物为主，而且还是最金贵的书画。国内能主要依靠自身馆藏搞这样一个展的博物馆恐怕不多。辽博却能，底气何来转年2020年12月，山高水长唐宋八大家主题文物展又在辽博开展，仍旧是书画为主，这次更是唐上加宋。辽博都藏有哪些唐宋法书图绘这要从辽博历史中寻找答案。1912年，溥仪退位后仍居住在紫禁城，也仍旧掌握着紫禁城收藏的历代文物珍玩。此时，溥仪以赏赐溥杰其弟的名义，由溥杰将大量书画作品盗出紫禁城。后溥仪出宫寓居天津，这批珍宝也随他辗转天津，其间为筹措复辟资金出卖了一部分珍宝。伪满洲国建立，珍宝又来到长春。1945年，日本投降，伪满洲国覆灭，溥仪仓促出逃，长春伪满皇宫中遗留下大量书画文物。北京天津乃至上海的古玩商闻风而动，纷纷麋集长春。1948年东北解放，对原国立沈阳博物馆进行整修，成立了东北博物馆。1949年7月，东北博物馆开放，成为新中国第一座博物馆。东北博物馆先接收了东北银行存放的一批重要书画作品，而后以杨仁恺为代表的东北博物馆专家，更是不断寻访散落民间的伪满皇宫文物。1959年，东北博物馆更名为辽宁省博物馆。当许多博物馆对明清书画作品还在孜孜以求时，辽博收藏的唐宋名迹，已可以与故宫上海博物馆等比肩。回到又见大唐，其中最重要的两幅画作为簪花仕女图和虢国夫人游春图。簪花仕女图绘有5名贵族妇女和1名侍女，另有两只狮子狗1只仙鹤。画面设色明艳线条圆劲，人物体态丰腴衣着华丽表情悠闲。此图被认定为唐代人物画家周昉所绘，现代对这个结论尚存疑问。但不论何人所绘，此图艺术性和对研究唐代绘画服饰妆容的价值都极高，并且可能是唯一传世的唐代仕女画作品，是绝对的艺术珍品。相对于簪花仕女图的断代争议，虢国夫人游春图则普遍被认为是宋人摹唐张萱作品。金章宗完颜璟更认为摹绘者是宋徽宗。虢国夫人是唐玄宗宠妃杨玉环的三姐，因杨玉环得宠，其一家均成为当朝新贵，而又随安史之乱身死名灭。其骄奢淫逸之人生的一个片段，被记录在这幅传世名作中。有此两幅名画可以撑得起又见大唐。然而，辽博收藏远不止于此。在唐代，最受推崇的书法家是晋人王羲之。唐太宗为永远拥有王羲之的兰亭集序，不惜将它带入坟墓。武则天也是王羲之的忠实拥趸，当时，侍郎王方庆为羲之后人，藏有王氏历代祖先手迹数十封，其中不乏王羲之王献之王僧虔等著名书家的作品。他将这些作品呈进给武则天，在他心中也许这些作品是有去无回了。没想到的是，武则天命人摹写一份之后，竟然将原作还给王方庆。在这点上，武则天比太宗派萧翼从辩才和尚手中骗走兰亭序要光明正大得多。1000多年过去，王方庆收藏的原作已无从寻找，武则天制作的摹本王羲之一门书翰，虽然屡遭火患损毁大半，但还有10幅作品保存至今。精妙的摹本，也得以使今人可以探究书圣一家笔法堂奥。除了唐人作品，辽博也收藏着不少五代宋人名迹。五代至北宋初年，绘画从唐代以人物画为主转向山水画为主，以描绘南北方不同景色，又分为南北派别。南唐北苑副使董源是南派代表人物，常以江南实景入画。辽博藏有他绘制的夏景山口待渡图，描写江南夏日，江岸沙渚疏林远树。北派画家则以关仝李成等为代表。辽博收藏的李成茂林远岫图，同样的山水画，却是北方大山大水，气象森严，和董源画作风气大不相同。宋徽宗虽是政治上的失败者，却是艺术上的成功者。辽博收藏有他多幅作品，又以一书一画最为著名。一书为草书千字文。这件作品写在11米长的描金云龙纹纸上。写这件作品时，徽宗40岁，正是艺术成熟而又精力旺盛的时候。全卷千余字，字字爽利，汪洋恣肆，毫不懈怠，可与怀素草书千字文一较高下。一画是瑞鹤图卷。这幅画颇有传奇色彩，政和二年上元次夕公元1112年正月十六，十几只仙鹤飞鸣于宫殿上空，让信奉道教的宋徽宗大为开心，以为祥瑞之兆，于是将此情景描摹下来。画作打破传统花鸟画构图，仙鹤和殿宇各占画面一半，呈现出对称稳定感。而画中碧蓝天色，900年后观之仍然令人心旷神怡。只是仙鹤没有保佑宋徽宗江山万年，瑞鹤出现15年后，金兵攻陷汴梁，徽宗被掳到五国城坐井观天。因为特殊的历史机缘收藏大量书画珍品，让辽博馆藏超越一般以展现本地历史为主的地方博物馆，成为具有全国性意义的博物馆。当年全程参与伪满流失文物收集的辽博副馆长杨仁恺先生，撰写了国宝沉浮录，考究书画真伪的同时，回忆曾经的历史瞬间。文物的命运是一个国家命运的缩影，当如今观众在现代化的辽博新馆欣赏先人巨制之时，抚今追昔怎能不感慨万千。来源中国青年报客户端']]
    # news = [b[-1] for b in batches]
    # print('news:', news)

    # with ClusterRpcProxy(config_mq) as rpc:
    #     datasets = rpc.data_service.preprocess(news, 'ext')
    #     summaries = rpc.summary_service.ext_summarize(datasets)
    # print('summaries:', summaries)
    # for i, b in enumerate(batches):
    #     insert_mysql(b[0], b[1], 'ext', b[2], summaries[i], b[3])
    # lock = Lock()
    # for i in range(2):
    #     p = Process(target=w, args=(lock,))
    #     p.start()
    # analysis()

# from torch import nn

# def output2words(ids, tokenizer, src_oovs):
#     words = []
#     for i in ids:
#         if i < tokenizer.__len__():
#             w = ''.join(tokenizer.convert_ids_to_tokens([i], skip_special_tokens=True))
#         else:
#             w = src_oovs[i - tokenizer.__len__()]
#         words.append(w)
#     return words


# def tokenize(text, tokenizer):
#     split_tokens = []
#     for token in jieba.cut(text, HMM=False):
#
#         if tokenizer.vocab.__contains__(token):
#             split_tokens.append(token)
#         else:
#             tokens = tokenizer.tokenize(token)
#             if not tokens.__contains__('[UNK]'):
#                 split_tokens.extend(tokens)
#             else:
#                 print(token)
#                 split_tokens.extend(token)
#             # if re.match('[a-zA-Z0-9 ]+', token):
#             #     split_tokens.extend(tokenizer.tokenize(token))
#             # else:
#             #     split_tokens.extend(list(token))
#     return split_tokens
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.l1 = nn.Linear(2, 2)
#         self.l2 = nn.Linear(2, 1)
#         self.l3 = nn.Linear(2, 1)
#     def forward(self, x):
#         x = self.l1(x)
#         y = self.l2(x)
#         z = self.l3(x)
#         return y, z

# analysis()
# input_text = '一辆小轿车，一名女司机，竟造成9死24伤。日前，深圳市交警局对事故进行通报：从目前证据看，事故系司机超速行驶且操作不当导致。目前24名伤员已有6名治愈出院，其余正接受治疗，预计事故赔偿费或超一千万元。'
# print(len(input_text))

# res = ext_summarize(input_text)
# print(platform.system())
# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
# t = tokenizer.tokenize('掮客')
# print('源文本：', '掮客')
# print('输出：', t)
# model = Model()
# x = torch.ones(2, 2)
# y, z = model(x)
# y = torch.abs(y)
# z = torch.abs(z)
# r = torch.ones(2, 1)
# loss1 = nn.BCELoss(reduction='none')(y, r).sum()
# loss2 = nn.BCELoss(reduction='none')(z, r).sum()
# loss = loss1 + loss2
# loss.backward(retain_graph=True)
# gygw = torch.autograd.grad(loss, list(model.l1.parameters()), retain_graph=True)
# print(gygw)
# print(2e-3)
# model = nn.Linear(1, 1)
# print(model)
# print(model.parameters())
# pass
# def statistic_dataset_information(model='mt5', raw_path='../json_data/'):
#     if model == 'mt5':
#         tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
#     else:
#         tokenizer = BertTokenizer.from_pretrained('../bert_base_chinese/', do_lower_case=True)
#     vocab = set()
#     src_tokens_len, tgt_tokens_len = 0, 0
#     jobs_len = 0
#     datasets = ['train', 'valid', 'test']
#     for corpus_type in datasets:
#         print('正在加载' + corpus_type + '数据集')
#         for json_file in glob.glob(join(raw_path, '*' + corpus_type + '_*.json')):
#             print('正在处理' + json_file.split('\\')[-1] + '文件。。。')
#             jobs = json.load(open(json_file, encoding='utf-8'))
#             jobs_len += len(jobs)
#             print('当前jobs长度:', len(jobs))
#             for i, d in enumerate(jobs):
#                 source, tgt = d['src'], d['tgt']
#                 if i % 1000 == 0:
#                     print(json_file.split('\\')[-1], '-------------------------------', i, '/',
#                           (i / len(jobs) * 100), '%')
#                 src_tokens = tokenize(''.join(source), tokenizer)
#                 tgt_tokens = tokenize(''.join(tgt), tokenizer)
#                 vocab.update(src_tokens + tgt_tokens)
#                 src_tokens_len += len(src_tokens)
#                 tgt_tokens_len += len(tgt_tokens)
#
#         print(corpus_type + '数据集文本平均tokens数量：' + str(int(src_tokens_len / jobs_len)))
#         print(corpus_type + '数据集摘要平均tokens数量：' + str(int(tgt_tokens_len / jobs_len)))
#         src_tokens_len, tgt_tokens_len = 0, 0
#         jobs_len = 0
#     print('词汇表大小：' + str(len(vocab)))
#
#
# statistic_dataset_information(model='bert')

# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
# model = MT5Model.from_pretrained('../t5_pegasus_chinese/')
#
# input_ids = tokenizer("研究表明养狗对你有好处",
#                       return_tensors="pt").input_ids  # Batch size 1
# decoder_input_ids = tokenizer("研究表明", return_tensors="pt").input_ids  # Batch size 1
#
# # forward pass
# outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
# last_hidden_states = outputs.last_hidden_state
#
# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
# model = MT5ForConditionalGeneration.from_pretrained('../t5_pegasus_chinese/')
#
# # training
# input_ids = tokenizer('<extra_id_0>走进<extra_id_1>公园', return_tensors='pt').input_ids
# labels = tokenizer('<extra_id_0>可爱的狗<extra_id_1>这个<extra_id_2>', return_tensors='pt').input_ids
# outputs = model(input_ids=input_ids, labels=labels)
# loss = outputs.loss
# logits = outputs.logits
#
# # inference
# input_ids = tokenizer("summarize: 研究表明养狗对你有好处",
#                       return_tensors="pt").input_ids  # Batch size 1
# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# studies have shown that owning a dog is good for you.

# document = ("在一辆客车上发生的疑似炸弹袭击中，至少有两人死亡",
#             "军方表示，周一在饱受冲突蹂躏的菲律宾南部。")
# # encode input context
# input_ids = tokenizer(document, return_tensors="pt", padding=True, truncation=True).input_ids
# print(input_ids)
# # generate 3 independent sequences using beam search decoding (5 beams)
# # with T5 encoder-decoder model conditioned on short news article.
# outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1)
# print(outputs)
# print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

# n_sents = [sum(i) for i in torch.tensor([[True, True, True, True, True, True, False]])]
# print(n_sents)
# src = (0, 1)
# print(src[1])

# def __merge_symmetry(sentences, symmetry=('“', '”')):
#     """合并对称符号，如双引号"""
#     effective_ = []
#     merged = True
#     for index in range(len(sentences)):
#         if symmetry[0] in sentences[index] and symmetry[1] not in sentences[index]:
#             merged = False
#             effective_.append(sentences[index])
#         elif symmetry[1] in sentences[index] and not merged:
#             merged = True
#             effective_[-1] += sentences[index]
#         elif symmetry[0] not in sentences[index] and symmetry[1] not in sentences[index] and not merged:
#             effective_[-1] += sentences[index]
#         else:
#             effective_.append(sentences[index])
#
#     return [i.strip() for i in effective_ if len(i.strip()) > 0]
#
#
# def to_sentences(paragraph):
#     """由段落切分成句子"""
#     sentences = re.split(r"(？|。|！|!|\?|…)", paragraph)
#     sentences.append("")
#     sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]  # 2是步数
#     sentences = [i.strip() for i in sentences if len(i.strip()) > 0]
#
#     for j in range(1, len(sentences)):
#         if sentences[j][0] == '”':
#             sentences[j - 1] = sentences[j - 1] + '”'
#             sentences[j] = sentences[j][1:]
#
#     return __merge_symmetry(sentences)
#     # return sentences
#
#
# src = '过去一年,陈德铭遇到了很多过去工作中不曾有的问题。“现在我们下去,各单位的领导还要见一见你,但是名分很不正。其实我最近一直在下面调研,卖了老面子,省委书记省长请我吃饭,但是我心里想,我是啥身份下来,我想不清楚。”他说。'
# print('src:', src)
# src = to_sentences(src)
# # print(src)
# for i in src:
#     print(i)

# def _compute_loss(batch, output, target):
#     # 抽取层
#     loss1 = torch.nn.BCELoss(reduction='none')(output[0], batch.labels.float())
#     loss1 = (loss1 * batch.mask_cls.float()).sum()
#
#     # # 生成层
#     # scores = _bottle(output[1])
#     # gtruth = target.contiguous().view(-1)
#     #
#     # loss2 = criterion(scores, gtruth)
#     # stats = _stats(loss2.clone(), scores, gtruth)
#
#     return loss1
#
#
# def _make_shard_state():
#     return {
#                "output": [[1, 2, 3],
#                           [1, 2, 3],
#                           [1, 2, 3],
#                           [1, 2, 3]],
#                "labels": [[2, 3, 4],
#                           [2, 3, 4],
#                           [2, 3, 4],
#                           [2, 3, 4]]
#            }, {
#                "output": [[5, 6, 7],
#                           [5, 6, 7],
#                           [5, 6, 7],
#                           [5, 6, 7]],
#                "target": [[6, 7, 8],
#                           [6, 7, 8],
#                           [6, 7, 8],
#                           [6, 7, 8]],
#            }
#
#
# def monolithic_compute_loss():
#     shard_state_ext, shard_state_abs = _make_shard_state()
#     res = [shard_state_ext['output'], shard_state_abs['output']]
#     aaa = [[a, b] for a, b in zip(shard_state_ext['output'], shard_state_abs['output'])]
#     # print(a, b)
#     loss1, _, batch_stats_abs = _compute_loss(**zip(shard_state_ext, shard_state_abs))
#
#     batch_stats_ext = reporter_ext.Statistics(float(loss1.cpu().data.numpy()), 2)
#
#     return batch_stats_ext, batch_stats_abs
#
# monolithic_compute_loss()

# args = argparse.Namespace(accum_count=2, alpha=1.0, batch_size=3000, beam_size=3, copy=True, corpora='NLPCC',
#                           decoder='mt5', load_from_extractive='', max_length=100, max_pos=1500, min_length=3,
#                           no_repeat_ngram_size=3, test_from='../models/mt5abs_nlpcc_cls/model_step_96000.pt',
#                           visible_gpus='-1', encoder='', extractor='', ext_ff_size=1, ext_heads=1, ext_dropout=1,
#                           ext_layers=1, param_init=1, param_init_is_xavier=1)
#
# device = "cpu" if args.visible_gpus == '-1' else "cuda"
# summarizer = ExtAbsSummarizer(args, device)
# print(summarizer)
# src = torch.rand(1, 10)
# src = torch.tensor([[0.1022, 0.5600, 0.0403, 0.2663, 0.8347, 0.4460, 0.4037, 0.0161, 0.6275, 0.2563]])
# print(src)
# clss = torch.tensor([[1, 4]])
# src1 = src[torch.arange(src.size(0)).unsqueeze(1), clss]
# src = src - src1
# print(src)
# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
# sentences_prefix = '句子：“'
# sentences_suffix = '”的意思是[MASK]'
# prompts_prefix = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences_prefix))
# prompts_suffix = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences_suffix))
# print(prompts_prefix, prompts_suffix)

# src = [
#     [1, 7423, 4619, 618, 3915, 4604, 309, 43836, 1218, 29571, 117, 8399, 520, 2334, 333, 178, 194, 2334, 333, 23224,
#      20063, 3399, 10884, 20584, 346, 10567, 117, 1626, 17792, 14645, 3399, 18432, 179, 19443, 30421, 117, 615,
#      17065, 346, 2335, 6228, 21232, 3399, 20584, 10594, 179, 12544, 117, 43836, 30064, 2335, 12608, 19210, 223,
#      26391, 178, 36479, 3399, 23224, 27336, 31802, 7155, 117, 18566, 24985, 175, 13726, 23224, 27333, 9802, 176,
#      117, 365, 39744, 17809, 117, 4920, 6257, 5034, 39127, 23224, 20063, 1463, 9739, 19772, 179, 2072, 4619, 618,
#      2243, 1979, 126, 2334, 5694, 2243, 23750, 117, 1101, 126, 2334, 5683, 2243, 7423, 4054, 2902, 3915, 12572,
#      3399, 7423, 28786, 18631, 19041, 30535, 198, 117, 2335, 12139, 23224, 2184, 5370, 2229, 2458, 10977, 3399,
#      8322, 30064, 117, 8399, 1625, 223, 520, 2334, 333, 1912, 194, 2334, 333, 17242, 2335, 10884, 3399, 20584,
#      10567, 117, 346, 17792, 14645, 3399, 18432, 179, 19443, 30421, 117, 615, 17065, 3399, 20842, 117, 17242, 2335,
#      6228, 21232, 3399, 20584, 10594, 179, 12987, 29653, 3399, 27794, 257, 17816, 117, 257, 7040, 21365, 1232, 5529,
#      117, 38798, 7005, 43688, 117, 28438, 2335, 6228, 33229, 309, 30653, 17406, 3399, 31931, 117, 11409, 25966,
#      3399, 28777, 117, 36100, 20093, 117, 23165, 25966, 28777, 346, 10605, 17816, 33335, 179, 26028, 4566, 23952,
#      117, 31486, 257, 4566, 784, 19942, 179, 2335, 23750, 3624, 117, 39621, 23224, 18155, 20063, 3399, 18107, 23774,
#      26391, 38340, 198, 223, 199, 2883, 799, 34217, 3399, 45265, 873, 8536, 20093, 117, 11409, 19218, 12744, 33707,
#      3399, 20144, 10376, 14226, 873, 27316, 14226, 117, 9464, 12744, 6957, 18138, 3399, 31197, 873, 8565, 32049,
#      117, 12845, 8737, 323, 7645, 2726, 335, 1101, 6957, 34217, 5080, 3399, 10703, 873, 32854, 179, 2335, 26326,
#      8322, 30064, 117, 23224, 11372, 257, 2274, 32363, 17435, 117, 22673, 505, 19377, 16180, 1119, 469, 23905, 758,
#      6756, 28624, 12795, 3705, 20579, 2374, 32080, 179, 2072, 7914, 117, 13726, 28786, 1487, 32425, 46857, 3399,
#      20586, 39356, 266, 8091, 27204, 6302, 30255, 3399, 8565, 117, 30421, 810, 15565, 34156, 16935, 23224, 11372,
#      17395, 24800, 117, 4920, 8091, 8565, 11409, 131, 7414, 26391, 178, 7409, 31200, 178, 25200, 2050, 1906, 178,
#      19490, 26391, 8506, 1632, 2678, 28786, 179, 20495, 275, 1686, 24800, 7947, 15249, 28780, 29652, 25876, 117,
#      4920, 1463, 39127, 18583, 16935, 23224, 11372, 12139, 7596, 20311, 1463, 1101, 25170, 13539, 7697, 3959, 443,
#      179, 1101, 23224, 27336, 20595, 117, 22484, 23224, 20063, 257, 23774, 13726, 27333, 9802, 26090, 30487, 117,
#      37727, 257, 1101, 14086, 873, 26391, 8838, 535, 34245, 30906, 117, 13050, 12737, 18432, 179, 257, 2335, 43836,
#      241, 1218, 117, 13738, 27336, 9802, 3399, 9786, 682, 6967, 28427, 175, 142, 116, 149, 451, 1215, 176, 3399,
#      17715, 179, 26026, 223, 26391, 178, 36479, 1101, 142, 4011, 178, 149, 4011, 16799, 198, 1112, 2335, 198, 1590,
#      519, 792, 16998, 117, 32638, 9802, 7709, 7155, 15449, 32821, 179, 15249, 44269, 32080, 28209, 178, 20322, 117,
#      2242, 28780, 230, 13738, 27336, 9802, 4093, 2048, 588, 198, 1590, 519, 792, 16998, 31200, 117, 257, 39841, 223,
#      26391, 178, 36479, 20641, 7155, 21187, 27793, 36361, 179, 1401, 430, 30571, 37255, 30064, 117, 43164, 1625,
#      23224, 20063, 3399, 32077, 1463, 19772, 117, 32833, 1463, 20087, 6785, 34217, 179, 9992, 1105, 3024, 8565,
#      8506, 12139, 23659, 23224, 20026, 3399, 46045, 21298, 21321, 29570, 12081, 32057, 516, 1107, 618, 179, 32020,
#      7671, 5034, 230, 42385, 21282, 26476, 26391, 6785, 34217, 3399, 14023, 9354, 266, 29142, 3399, 33103, 179, 142,
#      4011, 223, 7423, 26368, 178, 7423, 26391, 178, 23470, 23224, 178, 36477, 2337, 3705, 23224, 20063, 40090, 117,
#      12958, 9284, 32845, 9947, 179, 188, 26090, 7120, 189, 10884, 20584, 2342, 1625, 223, 10567, 23224, 20063, 1463,
#      9739, 19772, 2],
#     [1, 7423, 4619, 618, 3915, 4604, 309, 43836, 1218, 29571, 117, 8399, 520, 2334, 333, 178, 194, 2334, 333, 23224,
#      20063, 3399, 10884, 20584, 346, 10567, 117, 1626, 17792, 14645, 3399, 18432, 179, 19443, 30421, 117, 615,
#      17065, 346, 2335, 6228, 21232, 3399, 20584, 10594, 179, 12544, 117, 43836, 30064, 2335, 12608, 19210, 223,
#      26391, 178, 36479, 3399, 23224, 27336, 31802, 7155, 117, 18566, 24985, 175, 13726, 23224, 27333, 9802, 176,
#      117, 365, 39744, 17809, 117, 4920, 6257, 5034, 39127, 23224, 20063, 1463, 9739, 19772, 179, 2072, 4619, 618,
#      2243, 1979, 126, 2334, 5694, 2243, 23750, 117, 1101, 126, 2334, 5683, 2243, 7423, 4054, 2902, 3915, 12572,
#      3399, 7423, 28786, 18631, 19041, 30535, 198, 117, 2335, 12139, 23224, 2184, 5370, 2229, 2458, 10977, 3399,
#      8322, 30064, 117, 8399, 1625, 223, 520, 2334, 333, 1912, 194, 2334, 333, 17242, 2335, 10884, 3399, 20584,
#      10567, 117, 346, 17792, 14645, 3399, 18432, 179, 19443, 30421, 117, 615, 17065, 3399, 20842, 117, 17242, 2335,
#      6228, 21232, 3399, 20584, 10594, 179, 12987, 29653, 3399, 27794, 257, 17816, 117, 257, 7040, 21365, 1232, 5529,
#      117, 38798, 7005, 43688, 117, 28438, 2335, 6228, 33229, 309, 30653, 17406, 3399, 31931, 117, 11409, 25966,
#      3399, 28777, 117, 36100, 20093, 117, 23165, 25966, 28777, 346, 10605, 17816, 33335, 179, 26028, 4566, 23952,
#      117, 31486, 257, 4566, 784, 19942, 179, 2335, 23750, 3624, 117, 39621, 23224, 18155, 20063, 3399, 18107, 23774,
#      26391, 38340, 198, 223, 199, 2883, 799, 34217, 3399, 45265, 873, 8536, 20093, 117, 11409, 19218, 12744, 33707,
#      3399, 20144, 10376, 14226, 873, 27316, 14226, 117, 9464, 12744, 6957, 18138, 3399, 31197, 873, 8565, 32049,
#      117, 12845, 8737, 323, 7645, 2726, 335, 1101, 6957, 34217, 5080, 3399, 10703, 873, 32854, 179, 2335, 26326,
#      8322, 30064, 117, 23224, 11372, 257, 2274, 32363, 17435, 117, 22673, 505, 19377, 16180, 1119, 469, 23905, 758,
#      6756, 28624, 12795, 3705, 20579, 2374, 32080, 179, 2072, 7914, 117, 13726, 28786, 1487, 32425, 46857, 3399,
#      20586, 39356, 266, 8091, 27204, 6302, 30255, 3399, 8565, 117, 30421, 810, 15565, 34156, 16935, 23224, 11372,
#      17395, 24800, 117, 4920, 8091, 8565, 11409, 131, 7414, 26391, 178, 7409, 31200, 178, 25200, 2050, 1906, 178,
#      19490, 26391, 8506, 1632, 2678, 28786, 179, 20495, 275, 1686, 24800, 7947, 15249, 28780, 29652, 25876, 117,
#      4920, 1463, 39127, 18583, 16935, 23224, 11372, 12139, 7596, 20311, 1463, 1101, 25170, 13539, 7697, 3959, 443,
#      179, 1101, 23224, 27336, 20595, 117, 22484, 23224, 20063, 257, 23774, 13726, 27333, 9802, 26090, 30487, 117,
#      37727, 257, 1101, 14086, 873, 26391, 8838, 535, 34245, 30906, 117, 13050, 12737, 18432, 179, 257, 2335, 43836,
#      241, 1218, 117, 13738, 27336, 9802, 3399, 9786, 682, 6967, 28427, 175, 142, 116, 149, 451, 1215, 176, 3399,
#      17715, 179, 26026, 223, 26391, 178, 36479, 1101, 142, 4011, 178, 149, 28618, 13941, 1112, 2335, 198, 1590, 519,
#      792, 16998, 117, 32638, 9802, 7709, 7155, 15449, 32821, 179, 15249, 44269, 32080, 28209, 178, 20322, 117, 2242,
#      28780, 230, 13738, 27336, 9802, 4093, 2048, 588, 198, 1590, 519, 792, 16998, 31200, 117, 257, 39841, 223,
#      26391, 178, 36479, 20641, 7155, 21187, 27793, 36361, 179, 1401, 430, 30571, 37255, 30064, 117, 43164, 1625,
#      23224, 20063, 3399, 32077, 1463, 19772, 117, 32833, 1463, 20087, 6785, 34217, 179, 9992, 1105, 3024, 8565,
#      8506, 12139, 23659, 23224, 20026, 3399, 46045, 21298, 21321, 29570, 12081, 32057, 516, 1107, 618, 179, 32020,
#      7671, 5034, 230, 42385, 21282, 26476, 26391, 6785, 34217, 3399, 14023, 9354, 266, 29142, 3399, 33103, 179, 142,
#      4011, 223, 7423, 26368, 178, 7423, 26391, 178, 23470, 23224, 178, 36477, 2337, 3705, 23224, 20063, 40090, 117,
#      12958, 9284, 32845, 9947, 179, 188, 26090, 7120, 189, 10884, 20584, 2342, 1625, 223, 10567, 23224, 20063, 1463,
#      9739, 19772, 2]]

# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
# src = '中石化在A股、H股市场上均有上市公司平台'
# print(src)
# print(tokenize(src, tokenizer))
# print(tokenizer.tokenize(src))
# print(tokenizer.convert_tokens_to_ids(tokenize(src, tokenizer)))
# print(src2ids(src, tokenizer)[0])
# print(tokenizer.decode(src[0]))
# print(output2words(src[1], tokenizer, []))
# print(tokenizer.decode(src[1]))
# model = MT5EncoderModel.from_pretrained('../t5_pegasus_chinese/')
# print(model)
# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
# src = '◎每经记者刘明涛面对如今市场,在促销之外,商家还需做得更多。上周,北京现代宣布,其已下调销售目标,并降低批发量,减少经销商库存,促进销售。捷豹路虎也明确将降低经销商销售目标,并给予其额外补贴。当降价促销的大招放出,却仍不足以救市之时,调整产销目标、优化经销商考核标准成为更多车企的自救手段。单纯降价绕不开困境官降难见成效官降显然没有达到车企想要的效果。上周,全国乘用车市场信息联席会(以下简称乘联会)发布数据称,今年6月份,全国广义乘用车累计销量为1429676辆,同比下降3.2%,环比下降9.2%。从各大车企的半年成绩来看,上海大众虽以93.88万辆的成绩夺得冠军,且大幅领先于一汽大众和上汽通用,但同比去年却下滑了0.2%。排名第二、第三的一汽大众、上汽通用今年上半年销量分别为80.41万辆和78.91万辆,较去年同期相比,一汽大众下跌了11.3%,跌幅在前十企业中最高,上汽通用也有4.1%的下滑。不过,在6月销量榜单上,上汽通用以13.76万辆的销量,终结了上海大众“六连冠”之路。首开官降之举的上海大众,6月销量为13.09万辆,同比下滑了2.6%;一汽大众同比下滑达29.5%;而北京现代则下滑30.6%。'
# src = tokenize(src, tokenizer)
# print(src)
# print(src[torch.tensor(1): torch.tensor(3)])

# from sklearn.cluster import MiniBatchKMeans, KMeans
# import numpy as np
#
# X = np.array([[[1, 2], [1, 4], [1, 0],
#               [4, 2], [4, 0], [4, 4]],
#               [[4, 5], [0, 1], [2, 2],
#               [3, 2], [5, 5], [1, -1]]])
# # manually fit on batches
# kmeans = KMeans(n_clusters=2, random_state=0,
#                          )
# for i in X:
#     kmeans = kmeans.fit(i)
#     print(kmeans.cluster_centers_)
# kmeans = kmeans.partial_fit(X[6:12, :])

# os.environ['OMP_NUM_THREADS'] = '1'
# os.system("export OMP_NUM_THREADS")
# print(os.environ['OMP_NUM_THREADS'])

# src = "河南省平顶山市发布雷电黄色预警发布日期:2015-08-0315:17:00汝州市气象局2015年8月3日15时17分发布雷电黄色预警信号:我市大部分地区已出现雷电活动,局地伴有短时大风、强降水等对流性天气,且将持续,请注意防范!"
# doc_modified = re.sub(r':\w+:', "", emoji.demojize(src))
# print(doc_modified)

# src = [[[4, 3, 2], [1, 2, 4]],
#        [[5, 7, 1], [6, 2, 6]]]
# src = torch.tensor(src)
# print(src.shape)
# batch, sent, token = src.shape
# print(src)
# src = src.view(batch, -1)
# print(src)
# src = src.view(batch, sent, -1)
# print(src)

# src = np.round([[0.9999999, 0., 0., 0., 0.], [0.99997747, 0.9999988, 0., 0., 0.],
#        [0.99233854, 0.9632835, 0., 0., 0.], [0.9996667, 0.9473885, 0., 0., 0.],
#        [0.70036465, 0.9992889, 0.01224035, 0.00709263, 0.9588369]])
# print(src)
# src = [[i for i, v in enumerate(b) if v == 1] for b in src]
# print(src)

# src = torch.tensor([1])
# if src.size(1) == 1:
#     src = src.unsqueeze(-1)
# print(src.dim())
# print(src)

# rouge = test_rouge('../temp/', '../results/oracle_nlpcc_step0.raw_src',
#                    '../results/oracle_nlpcc_step0.gold')
# print(rouge_results_to_str(rouge))

# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
# data = {
#     "src": [
#         "今年是抗战胜利70周年。",
#         "自年初开始,互联网微博上就有许多贬抑共产党敌战区战场的帖子。",
#         "最耸人听闻的一则谣言出现在6月底新浪微博。",
#         "有署名“小右派”的网友发了这样一则长微博:“日本公布了二战在华阵亡数据318883人。",
#         "死于共军之手:851人。",
#         "基本与中国社会科学院的数据吻合。",
#         "共军百团大战毙敌302人;平型关大捷毙敌167人;38年晋察冀秋季反围攻毙敌39人;39年冀南春季反扫荡毙敌37人;39年冀中冬季反扫荡毙敌27人;40年春季反扫荡毙敌11人;115师陆房突围毙敌16人。",
#         "共击毙日寇599人,加上小战斗,合计被共军杀死851人。",
#         "死者都有姓名年龄、家乡,部队、死亡地点、被谁所杀详细记录。",
#         "”这则微博一出台,诸多实名网络大V争相转载。",
#         "目前,这则谣言在网上阅读量百万次以上,转载量超过数万次。",
#         "如不说明真相以正视听,将来又会如同此前“日本人从没有轰炸过延安”一样,由“谣言”上升为“真相”。",
#         "由于工作需要,国防大学《战事》剧组,阅读了大量抗战文献,其中就有日本防卫厅防卫研究所战史室编写的六卷本《中国事变陆军作战史》、两卷本《华北治安战》、单行本的《长沙作战》、《河南作战》、《湖北作战》、《广西作战》、《香港作战》、三卷本《昭和十七、八年的中国派遣军》等等,此外,我们还阅读了日本侵略者在多方搜集东北抗联信息情报基础上于1936年撰写的两卷本《东北共匪之研究》,曾任日本华北方面军总司令后又任中国派遣军总司令的《冈村宁次回忆录》。",
#         "现从中摘录部分资料,以正视听。",
#         "1、共产党抗日武装每天都在战斗我们阅读日本人撰写的资料后,得出一个印象:共产党抗日武装每天都在战斗状态,都在袭扰日军。",
#         "《华北治安战》这部书,详细记载了日本侵略者与坚持在华北敌战区与之进行游击战的共产党抗日武装之间的反复“拉锯战”。",
#         "其中包括了1939年以来,日本华北方面军在冈村宁次指挥下,对共产党抗日武装先后发起的冀东、冀中(包括五一扫荡)、冀南、晋南、晋中、鲁西等地数十次所谓“肃正作战”,以及先后五次勾联指挥华北华中伪政权机关、伪军和特务针对八路军、新四军发起的所谓“治安强化运动。",
#         "”书中确认,“中共游击战”是一场“不分昼夜、连续不断、永无休止的战争”,是使日军“深陷泥潭里的浴血战争”。",
#         "兹举书中若干记载:华北方面军第一一0师团报告,在1938年8月——1939年10月一年多一点时间中,与共产党武装交战次数约为2250次,每日平均约5次(上卷第156页)。",
#         "日军第二十七师团报告,从1939年1月至1940年11月间,仅该师团出发讨伐次数大小合计29168次,讨伐战斗次数为2759次(上卷第278页)。",
#         "1941年1月的所谓“冬季肃正作战”,日军各部与中共交战次数,仅在1月份内就达1682次,每日约有五、六十次战斗(下卷第17页)。",
#         "2、共产党敌后抗战战果辉煌共产党领导的敌后抗战,尽管没有正面战场那样的大会战,但在正面战场几乎所有的会战都以失败告终的情况下,惟敌后抗战节节胜利,不仅陷日寇于“每日面对不可测的恐怖”这样的惶恐不安之中,而且“积小胜为大胜”,有效消灭了敌人有生力量。",
#         "即使是从日本人的记载中,敌后战场战果与正面战场大会战相比也毫不逊色。",
#         "冈村宁次回忆录中写道(第326页):“共军的确长于谍报(在其本国以内),而且足智多谋,故经常出现我小部队被全歼的惨状。",
#         "”日本史学家、1941年从军的藤原彰在《中国战线从军记》中证实了冈村宁次的说法。",
#         "在专门回忆了1942年其所在联队有一个小队遭遇八路军伏击,全军覆没的“事件”后,他随即写道:“八路军的战术是,如果看到日军拥有优势兵力就撤退回避,发现日军处于劣势时,就预设埋伏,全歼日本士兵,然后夺走他们的武器装备。",
#         "”“像这样表明八路军战术成功,日军疏忽大意的事例,在冀东地区特别多。",
#         "中国驻屯步兵第一联队也经常有小部队被八路军全歼的事例发生。",
#         "”《华北治安战》中还记载有“百团大战”期间,日军几个小部队被共产党武装全歼的案例:“从9月24日晨,榆社、辽县之间的各警备队(独立步兵第十三大队)及东潞路的小滩镇警备队,同时遭到共军急袭。",
#         "这是第一二九师第三八六、第三八五旅所属的共军精锐部队,其兵力约8000人。",
#         "榆社——辽县道路上的榆社、常家会、王景村、铺上、管头等地的警备队,虽尽力进行防御战斗,但终因寡不敌众,大半遭到全歼,战死约80人。",
#         "”(上卷第314页)“9月23日夜,各据点同时遭受共军急袭,各自孤军奋战。",
#         "东圈堡(当时也称东团堡)及三甲村的守备队虽奋勇战斗,但终为玉碎。",
#         "共军最后从两阵地撤退时,在墙上写下‘该阵地日军守备队打的勇敢’等字样而去。",
#         "”(上卷第316页)关于各方歼敌总数量方面,在拍摄《战事》过程中,我们特地找日方研究战史专家证实,所谓“共军总共毙敌851人”没有任何根据。",
#         "相反,任何人仅仅只要根据日本防卫厅现有资料中零散的数字,也会清楚那则谣言是多么荒唐!",
#         "需要说明的是,日本防卫厅在编撰系列战史时,关于战果部分,是根据战时日军各部上报给大本营的记录整理出来的。",
#         "但正如日本特务头子土肥原在回忆录中所承认的那样,日军各部都在夸大己方战果,抑减中国方面战果。",
#         "“大本营发表的统计数字相当可观,但其中70%是为了夸耀战果而增加的水分。",
#         "”(转引自《华北治安战》译序第2页)尽管如此,我们还是可以把国共歼敌数量进行一个简单对比。",
#         "先摘录《华北治安战》中日方报告的一些零散记载:“第一一0师团报告,1938年8月—1939年10月间(即日方对我发起的所谓‘第三期肃正作战’期间),师团阵亡者,为533人。",
#         "”(上卷第156页)“第二十七师团报告,从1939年1月至1940年11月肃正作战期间,我忠勇的官兵丧失了649人,负伤1378人,甚为遗憾。",
#         "”(上卷第278)“在此次作战(第二次冀中作战)中,虽未查明彼我全面的损失,但在第一军方面损失最大的是独立混成第四旅团,(根据旅团第二期晋中作战战斗详报)战死71名、负伤66名、失踪2名。",
#         "另据旅团战死名簿记载,从8月20日至12月3日在旅团战死的276名。",
#         "”(上卷第312页)“关于此次作战(指1940年9月23日——10月12日间日方发起的所谓‘察南南境反击作战’)彼我的损失,根据军的统计,仅我独立混成第二旅团方面我战死133人、生死不明31人。",
#         "”(上卷第315页)1942年6月的冀中三号作战,“我方战死161名,其中军官9名,伤323名,其中军官14名”(下卷第161页)1943年9月对中共抗日武装发起的所谓“冀东一号终期作战”,战事于11月中旬结束,日方报告说,“我方损失也较大,计战死221人,伤91人。",
#         "”(下卷P214页)。",
#         "另据日本防卫厅《中国事变陆军作战史》记载(第三卷第二分册第179页),仅1941年这一年,华北方面军各部与中共交战次数为17198次!",
#         "日方损失是战死2352人,负伤501人。",
#         "可能有读者仍然认为,战死2352人,与共产党宣称的毙敌数目仍然有较大差距,且无法与国民党正面战场相比。",
#         "那我们就进行一个对比。",
#         "第一次长沙会战与第二次长沙会战,均由国民党战将薛岳指挥,国民党方面声称每次歼敌都在4万至5万余人。",
#         "日本防卫厅战史室在《长沙会战》一书中有如下记载:“在第一次长沙会战中,没能给予重庆军以应有打击……我方损失竟达战死1591人,战伤4412人。",
#         "”(第214页)“第11军发表的第2次长沙作战的战果及我方损失如下……敌遗弃尸体28612具……我方损失战死1591人,(其中军官数108人)战伤4412人(其中军官数241人),死伤战马1766匹。",
#         "”(第215页)对比一下,两次长沙会战,毙敌数量均远少于同一部书中所记载的1941年中共抗日武装所击毙日军数量的2352人!",
#         "再作一个对比:1943年5月至8月间浙赣作战,日方作战部队为中国派遣军第十三军和第十一军,对手是国民党第九战区长官薛岳指挥的第五十八军、第七十九军和第四军。",
#         "日方在《昭和十七八年的中国派遣军》中记载日第十三军报告战果如下(上卷第170-171页。",
#         ")“第一阶段(5月15日至5月29日),国军遗弃尸体48151,我方战死281人。",
#         "第二阶段(5月30日至6月15日),国民遗弃尸体12180,我方战死484人。",
#         "第三阶段(6月16日至8月14日),国军遗弃尸体6048,我方战死442人。",
#         "第四阶段(8月15日至9月30日),国军遗弃尸体1351,日方战死77人。",
#         "”“总计国军遗弃尸体共64430具,日方战死人数1284人,其中军官76人。",
#         "”所记载第十一军报告战果如下(上卷第188页):“国军共遗弃尸体15758具,我方战死336人,其中军官22人。",
#         "”日方结论说(上卷第189页):“第九战区长官薛岳在赣江以东使用了三个军,但是,他将这三个军逐个投入,最后,被我军各个歼灭。",
#         "同时,当他发现战场是在赣江以东之后,他仍然坚持在赣江西岸保存兵力,以致使兵力未能在战场上集中。",
#         "”对比一下可发现,浙赣会战各阶段,国民党军毙敌数量,与1942年以来共产党抗日武装在任何一次反扫荡作战中日方所记载的毙敌数量在同一个量级!",
#         "如:日方记载,“对中共的察南南境反击作战中,仅我独立混成第二旅团方面我战死133人、生死不明31人。",
#         "”“冀中三号作战,我方战死161名,其中军官9名,伤323名”。",
#         "因此,在评价国共抗战战果上,国内知识界有人在故意使用“双重标准”:在评估共产党战绩时,用日军大本营资料,且用不完全材料;在评估国民党战绩时,用国民党当局资料。",
#         "而他们最害怕的,就是用一把尺子量:因为如果都用日军大本营资料时,国共两党抗战战绩如何,就将大白于天下!",
#         "3、敌后游击战功效不只是歼灭有生力量还要看到,共产党敌占区游击战固然要歼灭敌有生力量,但最大的功效是“心理战”:让敌人恐惧,让人民看到胜利的希望!",
#         "日本战史刊物《历史群像》2002年第10期也刊登一则日本老兵回忆录:“我和国民党军打过仗,也和八路军打过仗,论武器装备是国民党军好得多,但八路军极善运动,也就是说对战场的控制力极强,随时随地都会向你发动进攻。",
#         "和他们作战我们无时无刻不在紧张中。",
#         "作为战士我们更不愿和八路军交手。",
#         "……和国民党军打仗,敌人败了就一跑了之,我们可以放心大胆地追击,和八路军打仗,即使撤退,他们也会设下各种陷阱,我们决不敢掉以轻心。",
#         "”4、“兵民是胜利之本”由于共产党抗日武装深深扎根于人民,成功粉碎了日寇一次次疯狂扫荡并壮大了自己的力量。",
#         "对此,《华北治安战》有记载。",
#         "第一期晋中作战(第一次反击作战)之后,在总结失败教训时,日军独立混成第四旅团长片山中将回忆如下(上卷第311页):“八路军的工作已深入到居民当中,村民正如‘空室清野’的标语那样,几乎逃避一空不见踪影,并且好像曾经积极协助八路军。",
#         "因而在作战期间,日军的动向被详细地泄露给八路军,但在日本方面则对八路军的情报完全不明。",
#         "”5、简短结语:莫让疯狂迷失了双眼综上所述,仅仅根据日方极少部分部队很零散的参与所谓“肃正作战”时的战报,不包括共产党武装主动发起的攻击,消灭日军已经甚众。",
#         "不知何来“抗战八年共军击毙日军仅851人”的根据?",
#         "更让人匪夷所思的是,如此离谱的谣言,在互联网上不仅登堂入世,还得到众多公知、大V们争相点赞和转载。",
#         "足见有些人“为反对而反对”到了何等地步?",
#         "抗日战争是近现代以来中华民族团结一致抵抗外侮获得的一次伟大胜利。",
#         "纪念抗战胜利70年,本是再次宣示捐弃历史前嫌、共同努力振兴中华的一次契机,但一些公知、大V出于消解中国共产党执政合法性目的,大量制造和传播贬损中共敌后抗战谣言,其结果只能是重翻历史旧账,再次撕裂阶层鸿沟。",
#         "对此,我们不能不予以高度重视和坚决反击!"
#     ],
#     "tgt": [
#         "红旗文稿称共产党抗日武装最大功效是心理战,让敌人恐惧让人民看到希望;有人想消解中共合法性"
#     ]
# }
# source, tgt = data['src'], data['tgt']
# # 最耸人听闻的一则谣言出现在6月底新浪微博。
# # 满足大于min_src_ntokens的句子才会被选中
# idxs = [i for i, s in enumerate(source) if len(s) > 5]
#
# # 截取超过max_src_ntokens的部分的不要
# src = [source[i][:150] for i in idxs]
# src = src[:100]
#
# src_token_tokens = []
# src_token_len = 0
# for i, sent in enumerate(source):
#     sent_tokens = tokenizer.tokenize(sent)
#     # 文字转成vocab映射后的 token
#     # 限定最终长度
#     src_token_len += len(sent_tokens)
#     if src_token_len > 1024:
#         sent_token_idxs = sent_tokens[:(len(sent_tokens) - src_token_len + 1024)]
#         src_token_tokens.append(sent_token_idxs)
#         print(i)
#         break
#     src_token_tokens.append(sent_tokens)
# print(len(sum(src_token_tokens, [])))
# print(src_token_tokens)
#
# oracle_ids = greedy_selection(src_token_tokens, [tokenizer.tokenize(s) for s in tgt], 10)
# # oracle_ids = greedy_selection([list(''.join(s)) for s in src_token_tokens],
# #                               [list(s) for s in tgt], 10)
# # [12, 14, 32, 68, 70, 84]  [2, 5, 12, 14, 17, 21, 24, 25]
# # oracle_ids = greedy_selection([list(s) for s in source],
# #                               [list(s) for s in tgt], 10)
# # oracle_ids = greedy_selection([list(jieba.cut(s, HMM=False)) for s in source],
# #                               [list(jieba.cut(s, HMM=False)) for s in tgt], 10)
# print(oracle_ids)
#
# for i in oracle_ids:
#     print(source[i])

# src = torch.rand(2, 1, 4)
# src = torch.tensor([[[0.8857, 0.2820, 0.9170, 0.4411]],
#
#                     [[0.9282, 0.5003, 0.8825, 0.1721]]])
# print(src.shape)
# temp = src.repeat(3, 1, 1)
# print(temp)
# print(temp.shape)
# src = src.repeat(1, 3, 1)
# src = src.view(-1, 1, src.size(-1))
# print(src)
# print(src.shape)
# extra_vocab_size = 3
# extra_vocab_index = list(range(-1, -extra_vocab_size - 1, -1))
# print(src[:, :, -extra_vocab_size:])
# a = torch.tensor([4, 2.3])
# src = torch.tensor([0, 0.])
# src[src == 0] = float('-inf')
# print(src)
#
# print(softmax(src, -1))
# distribute = softmax(src, -1)
# distribute[torch.isnan(distribute)] = 0
# print(distribute)
# print(a * distribute)

# print(list(range(-1, -4, -1)))

# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
# rs = output2words([50001, 157, 323, 3213, 545, 7654, 8098, 12, 434, 50000, 100], tokenizer, ['你', '号'])
# print(rs.replace(' ', ''))

# src = '{!--PGC_VIDEO:{\"status\":0,\"thumb_height\":360,\"vid\":\"00e6be45c5ec49d69e76465634d7baa9\",' \
#       '\"vname\":\"Ki2CJ8Ye-rGukufPs4S~kg__(1).mp4\",\"sp\":\"toutiao\",\"thumb_width\":360,' \
#       '\"vu\":\"00e6be45c5ec49d69e76465634d7baa9\",\"duration\":47,\"thumb_url\":\"4741\\/7693990042\",' \
#       '\"video_size\":{\"high\":{\"h\":480,\"w\":480},\"normal\":{\"h\":360,\"w\":360}}}--} '
# str = '2015-02-090655新浪体育三巨头82分骑士轻松大胜湖人您的浏览器不支持video标签。'
# src = re.sub(r"(\{!--PGC_VIDEO:.*}--})?(您的浏览器不支持video标签)?", '', str)
# print(src)

# for f in glob.glob(join('../raw_data/', '*_test.json' or '*_train.json')):
#     print(f)

# src = [
#     "江苏省委常委、南京市委书记杨卫泽涉嫌严重违纪违法,目前正接受组织调查。",
#     "杨卫泽简历:杨卫泽,男,汉族,1962年8月生,江苏常州人。",
#     "1988年4月入党,1981年8月参加工作。",
#     "在职研究生学历,硕士学位。",
#     "1978年12月起,南京航务工程专科学校港口水工建筑专业学习;1981年8月起,省交通厅规划计划处办事员、科员、基建计划科副科长、科长(其间:1986年5月—1988年7月挂职任邳县加口乡乡长助理);1990年11月起,省交通厅规划计划处副处长;1993年1月起,省扬子大桥股份有限公司经理部经理;1994年8月起,省交通厅规划计划处处长;1996年9月起,省交通厅副厅长、党组成员;1998年4月起,省交通厅厅长、党组副书记;2000年1月起,省交通厅厅长、党组书记兼江苏高速公路集团公司董事长、江苏润扬大桥发展有限公司董事长、总经理;2000年12月起,苏州市委副书记;2001年1月起,苏州市委副书记、代市长、市长(其间:2004年6月—2004年9月参加中组部赴哈佛大学公共管理高级人才培训班学习);2004年11月起,无锡市委书记;2006年11月起,省委常委、无锡市委书记;2011年3月起,省委常委、南京市委书记。",
#     "十七大代表,十届全国人大代表,省十次、十一次党代会代表,十届、十一届省委委员,十一届省委常委,省十届、十一届人大代表。",
#     "(摘自南京市委网站)"
# ]
# tgt = [
#     "http://admin.bytedance.com/dc/core/articletag/?",
#     "group=257406889"
# ]
# doc_sent_list = [jieba.cut(s, HMM=False) for s in src]
# abstract_sent_list = [jieba.cut(s, HMM=False) for s in tgt]
#
#
# def _rouge_clean(s):
#     # 去掉除了数字字母汉字以外的字符，[\u4E00-\u9FFF]+$ 匹配简体和繁体
#     return re.sub(r'[^a-zA-Z0-9\u4E00-\u9FFF ]', '', s)
#
#
# max_rouge = 0.0
# # abstract = sum(abstract_sent_list, [])
# abstract = [_rouge_clean(' '.join(s)).split() for s in abstract_sent_list]
# abstract = sum(abstract, [])
# sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
# evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
# reference_1grams = _get_word_ngrams(1, [abstract])
# evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
# reference_2grams = _get_word_ngrams(2, [abstract])
#
# selected = []
# for s in range(3):
#     cur_max_rouge = max_rouge
#     cur_id = -1
#     for i in range(len(doc_sent_list)):
#         if i in selected:
#             continue
#         c = selected + [i]
#         candidates_1 = [evaluated_1grams[idx] for idx in c]
#         candidates_1 = set.union(*map(set, candidates_1))
#         candidates_2 = [evaluated_2grams[idx] for idx in c]
#         candidates_2 = set.union(*map(set, candidates_2))
#         rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
#         rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
#         rouge_score = rouge_1 + rouge_2
#         if rouge_score > cur_max_rouge:
#             cur_max_rouge = rouge_score
#             cur_id = i
#     if cur_id == -1:
#         print(selected)
#         break
#     selected.append(cur_id)
#     max_rouge = cur_max_rouge

# print(sorted(selected))

# src = '詹姆斯33分骑士大胜公牛1-12015-05-0709:27新浪体育显示图片您的浏览器不支持video标签。北京时间5月7日,骑士主场以106-91轻取公牛,将总比'
# # rs = re.sub(r"^(\d{4}-\d{1,2}-\d{4}:\d{1,2}).*您的浏览器不支持video标签。", '', src)
# rs = re.sub(r"(\d{4}-\d{1,2}-\d{4}:\d{1,2}).*您的浏览器不支持video标签", '', src)
# rs = re.sub(r"^。", '', rs)
# print(rs)

# src = [[1, 2],
#        [3, 4],
#        [3, 4]]
# src[0] = [1] + src[0]
# src[-1] = src[-1] + [2]
# print(src)
#

# tokenizer = BertTokenizer.from_pretrained('../bert_base_chinese/', do_lower_case=True)
# # model = MT5ForConditionalGeneration.from_pretrained('../t5_pegasus_chinese/')
# print(tokenizer.tokenize('呕心沥血'))

# cross_attentions = torch.tensor([[[[0.1454, 0.2702, 0.5128, 0.0036],
#                                    [0.2301, 0.2683, 0.6051, 0.4403],
#                                    [0.5177, 0.8739, 0.5325, 0.2543]],
#
#                                   [[0.8813, 0.8082, 0.2765, 0.9165],
#                                    [0.2698, 0.5012, 0.1305, 0.7447],
#                                    [0.5919, 0.6366, 0.7387, 0.5645]]]])
# print(cross_attentions)
# concat_attentions = torch.tensor([[0.4931, 0.2746]]).transpose(0, 1)
# cross_attentions = cross_attentions.transpose(1, -1)
# print(cross_attentions.shape)
# cross_attentions = torch.matmul(cross_attentions, concat_attentions).squeeze(-1)
# print(cross_attentions.shape)
# print(cross_attentions.transpose(1, 2))

# tensor([[[[0.3137],
#           [0.3552],
#           [0.3288],
#           [0.2534]],
#
#          [[0.1875],
#           [0.2699],
#           [0.3342],
#           [0.4216]],
#
#          [[0.4178],
#           [0.6057],
#           [0.4654],
#           [0.2804]]]])

# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
# model = MT5ForConditionalGeneration.from_pretrained('../t5_pegasus_chinese/')
#
# # training
# input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
# labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
# # print(input_ids)
# # print(labels)
# outputs = model(input_ids=input_ids, labels=labels)
# loss = outputs.loss
# logits = outputs.logits
#
# # inference
# input_ids = tokenizer.encode(["总结：研究表明养狗对你有好处", "总结：研究表明养狗有好处"])
#
# print(input_ids)
# print(tokenizer.convert_ids_to_tokens(input_ids[0]))
# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# studies have shown that owning a dog is good for you.
# doc_modified = '銭躶'
# doc_modified = Converter('zh-hans').convert(doc_modified)
# print(doc_modified)
# doc_modified = traditional2simplified(doc_modified)
# print(doc_modified)

# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
# tokenizer = BertTokenizer.from_pretrained('../bert_base_chinese/', do_lower_case=True)
# tgt = ["“高温天气”是指地级市以上气象主管部门所属气象台向公众发布的日最高气温35°C以上的天气。", " "]
# tgt_tokens_str = '[unused1] ' + ' '.join(
#     [' '.join(tokenizer.tokenize(sent)) for sent in tgt]) + ' [unused2]'
# tgt_tokens = tgt_tokens_str.split()
# tgt_token_idxs = tokenizer.convert_tokens_to_ids(tgt_tokens)
# print(tgt_token_idxs)
# rs = tokenizer.encode(' '.join(tgt))
# print(rs)
#
# print(tokenizer.bos_token)
# print(tokenizer.eos_token_id)
# print(tokenizer.convert_ids_to_tokens(101))

# def traditional2simplified(sentence):
#     '''
#     将sentence中的繁体字转为简体字
#     :param sentence: 待转换的句子
#     :return: 将句子中繁体字转换为简体字之后的句子
#     '''
#     sentence = Converter('zh-hans').convert(sentence)
#     return sentence

# src = [
#     [118, 118, 6086, 48562, 100, 47741, 131, 168, 107, 35902, 100, 36048, 107, 131, 107, 43366, 131, 120, 120, 5765,
#      119, 5701, 119, 5681, 119, 5701, 120, 163, 48822, 48579, 48552, 119, 6107, 49041, 49225, 119, 6159, 119, 6107,
#      119, 35700, 120, 149, 49091, 48550, 49000, 48936, 48769, 119, 154,
#      48663, 48551, 119, 35902, 136, 163, 49362, 134, 100, 111, 5856, 134, 5702, 111, 157, 48867, 49002, 49469, 134,
#      123, 111, 147, 7410, 48404, 178, 38049, 638, 18173, 1686, 5529, 233, 43946,
#      1626, 30521, 179, 1686, 5529, 233, 17635, 117, 1454, 32845, 13888, 4566, 3857, 192, 4573, 602, 117, 17611,
#      13824, 9019, 27326, 117, 7374, 14066, 17395, 31851, 117, 33417, 1101, 11447,
#      857, 4903, 1105, 692, 26038, 2128, 39186, 873, 24448, 18668, 179, 100, 1686, 5529, 233, 7623, 12572, 36705,
#      11713, 12286, 16670, 19697, 8645, 11235, 28062, 36705, 175, 11547, 275, 176,
#      30298, 1686, 5529, 233, 17635, 117, 4566, 517, 5368, 4318, 1414, 1266, 182, 36705, 11713, 12286, 30298, 27700,
#      183, 117, 16389, 1454, 16670, 8544, 873, 45689, 32080, 27795, 10744, 117,
#      29653, 615, 10008, 11732, 117, 20923, 42286, 178, 39744, 179, 4566, 27545, 21282, 25644, 11447, 5365, 34497,
#      11212, 32018, 22051, 873, 5471, 4566, 332, 648, 117, 19017, 26444, 10977,
#      2092, 14214, 178, 3324, 15461, 20144, 117, 26971, 19720, 26090, 16670, 179, 1454, 32845, 13888, 4566, 3857,
#      192, 4573, 602, 117, 17611, 13824, 9019, 27326, 117, 7374, 14066, 17395,
#      31851, 117, 33417, 1101, 11447, 857, 4903, 1105, 692, 26038, 2128, 39186, 873, 24448, 18668, 179, 4566, 19017,
#      19697, 5081, 1230, 5395, 3430, 873, 32845, 16670, 8544, 117, 19390, 19720,
#      8216, 36116, 178, 25354, 46267, 178, 8228, 11624, 31675, 6589, 32845, 34217, 25060, 27055, 179, 4566, 17672,
#      873, 33263, 27882, 117, 3614, 1386, 4613, 4619, 4947, 566, 4947, 1266,
#      34119, 117, 19443, 13871, 6336, 178, 17413, 6336, 178, 9421, 6336, 3399, 30255
#         , 10977, 21293, 19041, 24188, 30421, 117, 44286, 1454, 3638, 14215, 178, 4667, 27892, 178, 18372, 3288, 873,
#      11713, 12286, 3399, 32860, 8957, 179, 4566, 323, 3023, 1605, 5368, 178, 505
#         , 2264, 807, 5314, 117, 3623, 2386, 1670, 1500, 20063, 10859, 873, 30653, 26576, 179, 4566, 11235, 28062,
#      36705, 175, 11547, 275, 176, 30298, 873, 21267, 43819, 117, 12294, 1454, 36705
#         , 11713, 12286, 12756, 16670, 3399, 17543, 2030, 1457, 378, 3290, 179, 44286, 7089, 13207, 38416, 8957,
#      1686, 5529, 233, 30255, 117, 197, 26160, 873, 21267, 32648, 4566, 13992, 175, 197, 218, 197, 1414, 176, 117,
#      20249, 19204, 117, 2185, 1120, 507, 5314, 117, 19697, 45689, 4318, 1105, 3288,
#      2448, 117, 1963, 16670, 1966, 1414, 19012, 1966, 588, 22857, 179, 1668, 3673
#         , 471, 517, 16670, 26295, 21532, 117, 7602, 23161, 11719, 117, 12173, 47022, 179, 11234, 32827, 33335,
#      30906, 117, 44286, 7089, 13207, 38416, 3399, 8957, 117, 19776, 45689, 3399, 41070
#         , 42546, 117, 26503, 36705, 11713, 12286, 8399, 15784, 29142, 17410, 179, 5030, 5084, 5654, 178, 3181, 663,
#      178, 1725, 687, 4931, 873, 36705, 11713, 12286, 5406, 1457, 1466, 3837, 18541, 588, 1607, 346, 4603, 117,
#      21267, 32648, 873, 11732, 47073, 603, 1607, 346, 4603, 179, 291, 534, 13065, 274,
#      2780, 25846, 16987, 8338, 19401, 230, 29653, 182, 36705, 11713, 12286, 30298, 27700, 183, 117, 128, 2334, 5697,
#      2243, 117, 37429, 201, 40267, 13065, 274, 2780, 16987, 19401, 16670,
#      117, 32144, 126, 1625, 22630, 31699, 6336, 19401, 16987, 117, 34479, 7277,
#      799, 32322, 35108, 806, 179, 4920, 2274, 7277, 42377, 30304, 21221, 178, 16422, 21218, 3399, 19401, 16987,
#      32322, 179, 127, 2334, 130, 2243, 117, 37011, 178, 38049, 11940, 182, 36705,
#      11713, 12286, 30298, 27700, 183, 179, 31994, 117, 37429, 178, 40267, 26129, 12572, 16800, 178, 26158, 9675,
#      117, 10672, 15808, 32425, 266, 7277, 29653, 30298, 27700, 3399, 15777, 18432
#         , 179, 7089, 19409, 117, 19697, 36705, 11713, 12286, 117, 34119, 2274, 14099, 117, 16987, 2274, 9957, 117,
#      21532, 2274, 9153, 179, 7583, 117, 7277, 46693, 28516, 10567, 182, 9931, 13688, 36705, 11713, 12286, 42385,
#      19720, 291, 534, 16987, 8338, 12190, 19401, 3399, 18432, 183, 179, 18432, 19744,
#      117, 19401, 16987, 15787, 12190, 8553, 117, 6940, 23494, 11732, 28494, 117, 323, 19401, 11732, 16670, 7568,
#      117, 7645, 31040, 19697, 291, 534, 7277, 11713, 12286, 21267, 16670, 179,
#      39367, 117, 291, 534, 274, 2780, 16987, 12190, 19401, 7645, 28551, 197, 20595, 131, 122, 119, 28551, 32845,
#      32648, 117, 1454, 12286, 20063, 178, 8216, 178, 25132, 178, 26839, 178,
#      22840, 178, 30298, 3705, 32845, 32648, 274, 2780, 16987, 132, 123, 119, 28551,
#      32845, 13888, 117, 26280, 9095, 13888, 5230, 12763, 12286, 117, 309, 25644, 18968, 11447, 5365, 34497, 11212,
#      8544, 4876, 5081, 178, 2335, 32845, 801, 378, 5395, 3430, 178, 13894, 26141, 3399, 692, 12131, 274, 2780,
#      16987, 19401, 132, 124, 119, 28551, 7077, 18984, 8338, 12763, 8197, 117,
#      32553, 32845, 29935, 274, 2780, 7089, 117, 9095, 3991, 801, 2185, 524, 873, 34119, 23867, 12763, 179, 1101,
#      34479, 32322, 16987, 223, 117, 11447, 32322, 1487, 3816, 16987, 5678, 806, 178,
#      14291, 16987, 5789, 806, 178, 7077, 45579, 5680, 806, 117, 23184, 32322, 37639, 16987, 5683, 806, 117, 753,
#      1218, 3816, 16987, 5733, 806, 117, 259, 26850, 16987, 5701, 806, 117, 7077,
#      45579, 5680, 806, 179, 21702, 120, 14959, 20561, 21468, 25955, 120, 4611,
#      1695, 857, 1635, 1401, 7643, 120, 414, 2050, 2806, 28084, 120, 1316, 4793, 1568, 36705, 11713, 12286, 18631,
#      15777, 18368, 25849, 117, 230, 20138, 2347, 5329, 3876, 3905, 33435, 24448,
#      18668, 3023, 4774, 106],
#
#     [118, 118, 6086, 48562, 50000, 47741, 131, 168, 107, 35902, 50000, 36048, 107, 131, 107, 43366, 131, 120,
#      120, 5765, 119, 5701, 119, 5681, 119, 5701, 120, 163, 48822, 48579, 48552, 119
#         , 6107, 49041, 49225, 119, 6159, 119, 6107, 119, 35700, 120, 149, 49091, 48550, 49000, 48936, 48769, 119,
#      154, 48663, 48551, 119, 35902, 136, 163, 49362, 134, 128, 123, 122, 129, 143,
#      129, 122, 126, 130, 143, 129, 124, 130, 130, 146, 123, 145, 127, 126, 144, 123, 122, 128, 125, 122, 147,
#      143, 146, 147, 121, 130, 128, 147, 129, 124, 144, 143, 129, 145, 128, 122, 130,
#      147, 144, 123, 122, 125, 124, 127, 130, 122, 125, 142, 129, 147, 125, 126, 130, 127, 123, 123, 130, 122,
#      125, 129, 142, 143, 129, 130, 142, 128, 128, 127, 142, 125, 147, 123, 123, 147
#         , 130, 145, 144, 128, 130, 121, 129, 122, 130, 121, 126, 142, 130, 124, 145, 147, 145, 128, 126, 147, 125,
#      142, 127, 142, 130, 146, 128, 124, 128, 121, 143, 127, 122, 124, 145, 144, 121, 142, 143, 142, 124, 143, 144,
#      142, 147, 147, 147, 142, 147, 130, 127, 144, 145, 123, 126, 142, 125, 145,
#      144,
#      142, 145, 144, 129, 122, 127, 146, 125, 126, 129, 125, 130, 121, 128, 129, 144, 145, 143, 143, 146, 144, 125,
#      143, 121, 142, 143, 129, 123, 123, 126, 123, 143, 145, 142, 143, 121,
#      147, 122, 111, 5856, 134, 5702, 111, 157, 48867, 49002, 49469, 134, 123, 111, 147, 7410, 48404, 178, 38049,
#      638, 18173, 1686, 5529, 233, 43946, 1626, 30521, 179, 1686, 5529, 233,
#      17635,
#      117, 1454, 32845, 13888, 4566, 3857, 192, 4573, 602, 117, 17611, 13824, 9019, 27326, 117, 7374, 14066, 17395,
#      31851, 117, 33417, 1101, 11447, 857, 4903, 1105, 692, 26038, 2128,
#      39186,
#      873, 24448, 18668, 179, 50001, 1686, 5529, 233, 7623, 12572, 36705, 11713
#         , 12286, 16670, 19697, 8645, 11235, 28062, 36705, 175, 11547, 275, 176, 30298, 1686, 5529, 233, 17635, 117,
#      4566, 517, 5368, 4318, 1414, 1266, 182, 36705, 11713, 12286, 30298, 27700, 183, 117, 16389, 1454, 16670, 8544,
#      873, 45689, 32080, 27795, 10744, 117, 29653, 615, 10008, 11732, 117,
#      20923,
#      42286, 178, 39744, 179, 4566, 27545, 21282, 25644, 11447, 5365, 34497, 11212, 32018, 22051, 873, 5471, 4566,
#      332, 648, 117, 19017, 26444, 10977, 2092, 14214, 178, 3324, 15461,
#      20144,
#      117, 26971, 19720, 26090, 16670, 179, 1454, 32845, 13888, 4566, 3857, 192,
#      4573, 602, 117, 17611, 13824, 9019, 27326, 117, 7374, 14066, 17395, 31851, 117, 33417, 1101, 11447, 857,
#      4903,
#      1105, 692, 26038, 2128, 39186, 873, 24448, 18668, 179, 4566, 19017, 19697, 5081, 1230, 5395, 3430, 873, 32845,
#      16670, 8544, 117, 19390, 19720, 8216, 36116, 178, 25354, 46267, 178,
#      8228, 11624, 31675, 6589, 32845, 34217, 25060, 27055, 179, 4566, 17672, 873,
#      33263, 27882, 117, 3614, 1386, 4613, 4619, 4947, 566, 4947, 1266, 34119, 117, 19443, 13871, 6336, 178,
#      17413,
#      6336, 178, 9421, 6336, 3399, 30255, 10977, 21293, 19041, 24188, 30421, 117, 44286, 1454, 3638, 14215, 178,
#      4667, 27892, 178, 18372, 3288, 873, 11713, 12286, 3399, 32860, 8957, 179,
#      4566, 323, 3023, 1605, 5368, 178, 505, 2264, 807, 5314, 117, 3623, 2386, 1670, 1500, 20063, 10859, 873, 30653,
#      26576, 179, 4566, 11235, 28062, 36705, 175, 11547, 275, 176, 30298, 873,
#      21267, 43819, 117, 12294, 1454, 36705, 11713, 12286, 12756, 16670, 3399, 17543, 2030, 1457, 378, 3290, 179,
#      44286, 7089, 13207, 38416, 8957, 1686, 5529, 233, 30255, 117, 197, 26160,
#      873,
#      21267, 32648, 4566, 13992, 175, 197, 218, 197, 1414, 176, 117, 20249, 19204, 117, 2185, 1120, 507, 5314,
#      117, 19697, 45689, 4318, 1105, 3288, 2448, 117, 1963, 16670, 1966, 1414, 19012,
#      1966, 588, 22857, 179, 1668, 3673, 471, 517, 16670, 26295, 21532, 117, 7602, 23161, 11719, 117, 12173,
#      47022, 179, 11234, 32827, 33335, 30906, 117, 44286, 7089, 13207, 38416, 3399,
#      8957, 117, 19776, 45689, 3399, 41070, 42546, 117, 26503, 36705, 11713, 12286, 8399, 15784, 29142, 17410,
#      179, 5030, 5084, 5654, 178, 3181, 663, 178, 1725, 687, 4931, 873, 36705, 11713,
#      12286, 5406, 1457, 1466, 3837, 18541, 588, 1607, 346, 4603, 117, 21267, 32648, 873, 11732, 47073, 603, 1607,
#      346, 4603, 179, 291, 534, 13065, 274, 2780, 25846, 16987, 8338, 19401, 230,
#      29653, 182, 36705, 11713, 12286, 30298, 27700, 183, 117, 128, 2334, 5697,
#      2243, 117, 37429, 201, 40267, 13065, 274, 2780, 16987, 19401, 16670, 117, 32144, 126, 1625, 22630, 31699,
#      6336,
#      19401, 16987, 117, 34479, 7277, 799, 32322, 35108, 806, 179, 4920, 2274
#         , 7277, 42377, 30304, 21221, 178, 16422, 21218, 3399, 19401, 16987, 32322, 179, 127, 2334, 130, 2243, 117,
#      37011, 178, 38049, 11940, 182, 36705, 11713, 12286, 30298, 27700, 183, 179, 31994, 117, 37429, 178, 40267,
#      26129, 12572, 16800, 178, 26158, 9675, 117, 10672, 15808, 32425, 266, 7277, 29653,
#      30298, 27700, 3399, 15777, 18432, 179, 7089, 19409, 117, 19697, 36705,
#      11713, 12286, 117, 34119, 2274, 14099, 117, 16987, 2274, 9957, 117, 21532, 2274, 9153, 179, 7583, 117, 7277,
#      46693, 28516, 10567, 182, 9931, 13688, 36705, 11713, 12286, 42385, 19720, 291, 534, 16987, 8338, 12190,
#      19401, 3399, 18432, 183, 179, 18432, 19744, 117, 19401, 16987, 15787, 12190, 8553,
#      117, 6940, 23494, 11732, 28494, 117, 323, 19401, 11732, 16670, 7568, 117
#         , 7645, 31040, 19697, 291, 534, 7277, 11713, 12286, 21267, 16670, 179, 39367, 117, 291, 534, 274, 2780,
#      16987,
#      12190, 19401, 7645, 28551, 197, 20595, 131, 122, 119, 28551, 32845, 32648
#         , 117, 1454, 12286, 20063, 178, 8216, 178, 25132, 178, 26839, 178, 22840, 178, 30298, 3705, 32845, 32648,
#      274,
#      2780, 16987, 132, 123, 119, 28551, 32845, 13888, 117, 26280, 9095, 13888,
#      5230, 12763, 12286, 117, 309, 25644, 18968, 11447, 5365, 34497, 11212, 8544, 4876, 5081, 178, 2335, 32845,
#      801,
#      378, 5395, 3430, 178, 13894, 26141, 3399, 692, 12131, 274, 2780, 16987,
#      19401, 132, 124, 119, 28551, 7077, 18984, 8338, 12763, 8197, 117, 32553, 32845, 29935, 274, 2780, 7089, 117,
#      9095, 3991, 801, 2185, 524, 873, 34119, 23867, 12763, 179, 1101, 34479, 32322, 16987, 223, 117, 11447,
#      32322, 1487, 3816, 16987, 5678, 806, 178, 14291, 16987, 5789, 806, 178, 7077,
#      45579, 5680, 806, 117, 23184, 32322, 37639, 16987, 5683, 806, 117, 753, 1218,
#      3816, 16987, 5733, 806, 117, 259, 26850, 16987, 5701, 806, 117, 7077, 45579, 5680, 806, 179, 21702, 120,
#      14959,
#      20561, 21468, 25955, 120, 4611, 1695, 857, 1635, 1401, 7643, 120, 414,
#      2050, 2806, 28084, 120, 1316, 4793, 1568, 36705, 11713, 12286, 18631, 15777, 18368, 25849, 117, 230, 20138,
#      2347, 5329, 3876, 3905, 33435, 24448, 18668, 3023, 4774, 106]]
# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True, )
# src_oovs = ['_', '△']
# # print(tokenizer.decode(src[0]))
# print(tokenizer.decode(src[0]).replace(' ', ''))
# print(''.join(output2words(src[1], tokenizer, src_oovs)).replace('##', ''))
#
# src = '--pgc_video:{"mp4_url":"http://60.28.13.28/vhot2.qqvideo.tc.qq.com/h0160tcotei.m701.mp4?vkey' \
#       '=7218b8159b8399e2d65c21741fbef097f83cb8d719fc21436914a8f4596229148ab89a776a4f22f9dc79081905a93dfd75f4a6a9e73' \
#       '70b613dc0aba3bcafffaf96cd25a4dcadc816e45849078cdbbec4b0ab82252bdab0f1&br=29&platform=2&f' \
#       '中共中央政治局常委、国务院副总理张高丽主持会议并讲话。张高丽强调,对重点地区要统一规划,强化土地供应管控,严格城镇开发边界,' \
#       '防止在北京周边地区盲目搞房地产和炒作房价。△张高丽主持召开京津冀协同发展工作推动会议加快编制京津冀“十三五”规划张高丽强调,要全面落实好《京津冀协同发展规划纲要》,尽快对工作任务和政策措施进行细化分解,' \
#       '落实到具体单位,明确路线图、时间表。要紧扣有序疏解北京非首都功能这个核心和首要任务,抓紧研究制定控增量、疏存量政策,稳妥推进相关工作。对重点地区要统一规划,强化土地供应管控,严格城镇开发边界,' \
#       '防止在北京周边地区盲目搞房地产和炒作房价。要抓紧推动重大项目和重点工作任务,持续推进交通一体化、生态环境保护、产业升级转移三个重点领域率先突破。要当前和长远结合,科学论证选准选好项目,' \
#       '按照在建一批、开工一批、储备一批的要求制定有效投资滚动计划,充分发挥对稳增长、调结构、惠民生和协同发展的重要作用。要以点带面、先易后难,积极开展改革创新和试点示范。要加快编制京津冀“十三五”规划和有关专项规划,' \
#       '发挥对京津冀协同发展各项工作的引领指导作用。充分发挥专家咨询委员会作用张高丽要求,三省市和有关部门要坚持“三严三实”,敢于担当,攻坚克难,推动政策措施落地生根,把工作抓实抓好抓出水平。建立健全工作督办机制,' \
#       '主动沟通协调,及时解决问题。加强重大问题调研,充分发挥专家咨询委员会的作用,提高政策措施的科学性针对性,确保京津冀协同发展今年实现良好开局。郭金龙、王勇、徐匡迪和京津冀协同发展领导小组成员出席会议,' \
#       '有关部门和单位负责同志列席会议。京冀启动互派百名干部人才挂职为落实《京津冀协同发展规划纲要》,7月24日,北京市与河北省启动互派干部挂职工作,连续5年每年轮换一批挂职干部,' \
#       '首批两地各选派100名。这是两地近年来规模最大、层次最全的挂职干部选派。6月9日,党中央、国务院印发《京津冀协同发展规划纲要》。近日,北京市、河北省相继召开市委、省委全会,' \
#       '分别审议通过了两地落实规划纲要的实施意见。专家指出,推动京津冀协同发展,项目是基础,干部是关键,机制是保障。为此,两地组织部门联合出台《关于围绕京津冀协同发展进一步推进京冀干部人才双向挂职的意见》。意见提出,' \
#       '挂职干部实行双向任职,不免派出单位职务,以挂职单位工作为主,主要负责推动京冀两地协同发展有关工作。据介绍,京冀互派干部双向挂职主要聚焦三方面:1.聚焦重点部门,' \
#       '对发展改革、交通、环保、科技、水务、规划等重点部门互派干部;2.聚焦重点地区,着眼促进地区间合作发展,从疏解承接北京非首都功能任务较重、有重点合作项目、地域相邻的区县市互派干部挂职;3.聚焦专业技术人才合作交流,' \
#       '遴选重点行业互派专家,促进联合攻关和项目深度合作。在首批选派干部中,北京选派局级干部10名、处级干部78名、专业技术人员12名,河北选派厅局级干部15名,县处级干部45名,乡科级干部28名,' \
#       '专业技术人员12名。来源/央视新闻本期监制/许强周庆安主编/侯振海编辑/娄越巍京津冀协同发展战略实施惠及百姓,为政府未雨绸缪防范炒作房价点赞! '
# print(tokenizer(src).input_ids)
# print(src2ids(src, tokenizer))
#
# src = '7218b8159b8399e2d65c21741fbef097f83cb8d719fc21436914a8f4596229148ab89a776a4f22f9dc79081905a93dfd75f4a6a9e7370b613dc0aba3bcafffaf96cd25a4dcadc816e45849078cdbbec4b0ab82252bdab0f1 '
# rs = []
# rs.append(src)
# print(rs)

# print('\u200d' * 2, 1)
# print(1)
# str = '\u200d\u200d\u200d\u200d个人资料\u200d\u200d宋建国,男,北京市公安交通管理局局长。', '因涉嫌利用职务之便,在购车摇号工作中,存在徇私舞弊行为。'
# s = chr(8204) + chr(8205)
# print(s)
# print(repr(str))  # '\u200c\u200d'
#
# def remove_upprintable_chars(s):
#     """移除所有不可见字符"""
#     return ''.join(x for x in s if x.isprintable())
#
# s = 'A\u2029B'
# print(s.isprintable(), s)  # False
# s = remove_upprintable_chars(s)
# print(s.isprintable(), s)  # True

# start = time.time()
# # 截取超过max_src_ntokens的部分的不要
# src = [src[i][:200 - 1] + '。' for i in range(2)]
# print(src)

# src_extend_vocab, src_oovs = src2ids(src, tokenizer, do_lower_case=True)
# pred_sent = output2words(src_extend_vocab, tokenizer, src_oovs)
# print('花费时间：', (time.time() - start) * 5 * 200000 / 3600 / 24)
# print('花费时间：', (time.time() - start))

# doc_modified = re.sub(r' ', ",", src)
# 去掉NLPCC句内的无用内容
# doc_modified = re.sub(r'\(?组图\)?', "", doc_modified)
# doc_modified = re.sub(r'\(?现场图\)?', "", doc_modified)
# doc_modified = re.sub(r':\w+:', "", emoji.demojize(src))
# doc_modified = doc_modified.strip()  # 去空格
# # 去掉NLPCC句内的无用内容
# doc_modified = re.sub(r'发布日期:', "", doc_modified)
# doc_modified = re.sub(r'[0-9]+-[0-9]+-[0-9]+:?[0-9]+', "", doc_modified)
# doc_modified = re.sub(r'【字体:】', "", doc_modified)
# doc_modified = re.sub(r'显示图片', "", doc_modified)
# doc_modified = re.sub(r'其?[0-9一二三四五六七八九十]+[，、.]', "", doc_modified)
# print(doc_modified)

# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
# rs = tokenizer.convert_tokens_to_ids('[unused1]')
# print(rs)
# src = torch.rand(4, 4)
# def tokenize(text, tokenizer):
#     split_tokens = []
#     for token in jieba.cut(text, HMM=False):
#         if tokenizer.vocab.__contains__(token):
#             split_tokens.append(token)
#         else:
#             if re.match('[a-zA-Z0-9 ]+', token):
#                 split_tokens.extend(tokenizer.tokenize(token))
#             else:
#                 split_tokens.extend(list(token))
#     print(split_tokens)
#     return split_tokens
# print(tokenize('你是 傻逼！faf74 a5', tokenizer))

# src = torch.tensor([
#     [[-87.8186, -90.3359, -87.8186, -59.4182, -1059.9231, 1035.1404],
#      [-125.9021, -127.9688, -125.9021, -240.8048, -259.7015, -1637.0332]],
#
#     [[282.6374, 10.2367, 12.6374, -92.7165, -78319.3690, 286.8060],
#      [-35.8624, -35.2809, -35.8624, -117.0540, -779.5608, -1980.0664]]], dtype=torch.float).float()
# print(torch.softmax(src, dim=-1))

# start = time.time()
# input_ids = torch.tensor([[0, 1561, 753, 192, 15161, 10928, 35957, 2355, 15125, 35464,
#                            36848, 4534, 183, 10792, 117, 2331],
#                           [0, 1561, 753, 192, 15161, 10928, 35957, 2355, 15125, 35464,
#                            36848, 4534, 183, 10792, 179, 350],
#                           [0, 1561, 753, 192, 15161, 10928, 35957, 2355, 15125, 35464,
#                            36848, 4534, 183, 10792, 117, 43836],
#                           [0, 182, 293, 629, 183, 223, 3399, 175, 18632, 176,
#                            2363, 273, 5654, 12068, 3181, 4917],
#                           [0, 182, 293, 629, 183, 223, 3399, 175, 18632, 176,
#                            2363, 273, 5654, 32544, 28165, 350],
#                           [0, 182, 293, 629, 183, 223, 3399, 175, 18632, 176,
#                            2363, 273, 5654, 178, 175, 50000]], device='cuda:0')
# # print(input_ids[:, -1])
# # rs = input_ids[:, -1].masked_fill_(input_ids[:, -1] >= 50000, 101)  # convert oov to unk
# rs = input_ids.masked_fill_(input_ids >= 50000, 101)  # convert oov to unk
# print(input_ids)
# print('时间：', time.time() - start)
# 2.3125243186950684

# vocab = load_vocab('../t5_pegasus_chinese/vocab.txt')
# print(vocab)

# def yield_tokens(file_path):
#     with io.open(file_path, encoding='utf-8') as f:
#         print(f.read())
#         for line in f:
#             yield line.strip().split()
#
#
# vocab = build_vocab_from_iterator(yield_tokens('../t5_pegasus_chinese/vocab.txt'), specials=["[UNK]"])
# print(vocab.lookup_indices(['[UNK]']))
# print(vocab.lookup_token(100))
# print(vocab.__len__())

# tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)
# src = [[], ['北京'], ['宅'], ['地'], ['市场'], ['迎来'], ['马'], ['年'], ['首次'], ['出让'], ['丰台'], ['和'], ['顺义'], ['两'], ['宗'],
#        ['地'], ['累计'], ['出让金'], ['达'],
#        ['8'], ['亿元'], [','], ['其中'], ['丰台'], ['西'], ['局'], ['地块'], ['以'], ['7'], ['亿元'], ['拍'], ['出'], [','],
#        ['仅次于'], ['%'], ['有'], ['农'], ['展'], ['馆'], ['地块'], ['3']
#     , ['万元'], ['/'], ['平米'], ['的'], ['价格'], ['。'], ['竞拍'], ['企业'], ['表现'], ['明显'], ['。'], ['龙湖'], ['再次'], ['“'],
#        ['虎口'], ['夺'], ['食'], ['”'], ['。'], ['现场'], [
#            '花絮'], ['。'], ['任'], ['志'], ['强'], ['炮轰'], ['国土局'], ['。'], ['。'], ['不'], ['认'], ['身份证'], ['1'], ['复印件'],
#        ['。'], ['的'], ['炮轰'], [','], ['市'], ['天价'], ['国土局'], ['的'], [', '], ['已'], ['位居'], ['债券'], ['市场'], ['前'],
#        ['10'], ['2012'], ['显示'], ['的'], ['的'], ['债券'], [', '], [', '], ['都'], ['可观'], ['。']]
# src = [''.join(tt) for tt in src]
# print(src)
# print(''.join(src))
# vocab = load_vocab('../t5_pegasus_chinese/vocab.txt')
# src = '拮'
# tokens = tokenizer.tokenize('韩华·闹')
# print(tokens)
# tokens = tokenize(src, vocab)
# print(''.join(tokens).replace('##', ''))
# print(src)
# rs = [i for i in jieba.cut('韩华·闹', HMM=False)]
# print(rs)

# # str = 'NineWest'
# src = [[2269, 4714, 3915, 168, 106, 118, 118, 6086, 48562, 100,
#         47741, 131, 168, 107, 6152, 49662, 107, 131, 121, 117,
#         107, 36034, 48886, 100, 5968, 49738, 107, 131, 35320, 117,
#         107, 36034, 48886, 100, 36048, 107, 131, 107, 5773, 48692,
#         120, 5781, 48678, 48653, 48605, 48559, 107, 117, 107, 6189,
#         48563, 107, 131, 43477, 117, 107, 36034, 48886, 100, 6195,
#         48759, 48567, 107, 131, 35425, 117, 107, 6149, 107, 131,
#         107, 6165, 49032, 48830, 48574, 107, 117, 107, 162, 48970,
#         107, 131, 107, 35220, 48644, 120, 5749, 48663, 48644, 48681,
#         48607, 107, 117, 107, 163, 48580, 107, 131, 107, 5756,
#         48560, 126, 1510, 25563, 25102, 2834, 22603, 2132, 1456, 1370,
#         830, 269, 15186, 4606, 4625, 7944, 27868, 25960, 30330, 18643,
#         188, 32615, 1080, 15386, 21575, 15553, 189, 137, 42489, 4977,
#         131, 22111, 17676, 117, 627, 5284, 3399, 32615, 1963, 25102,
#         2834, 30796, 615, 15889, 117, 22568, 3399, 1463, 527, 2084,
#         2613, 179, 38440, 8186, 314, 15889, 257, 2335, 221, 15553,
#         117, 16923, 44209, 117, 10548, 1454, 25102, 2834, 3399, 15386,
#         1935, 199, 2604, 2637, 1934, 179, 32615, 23776, 2188, 1866,
#         2357, 297, 3923, 1581, 4534, 10792, 179, 2333, 6992, 26181,
#         3399, 18295, 32011, 12312, 266, 117, 15553, 117, 1874, 14827,
#         23165, 8390, 117, 23165, 25725, 188, 1779, 4863, 106, 23185,
#         20473, 6279, 126, 1510, 38654, 4534, 1982, 4785, 106, 189,
#         128, 2334, 129, 2243, 6842, 127, 3023, 4611, 117, 2228,
#         259, 753, 1466, 534, 5201, 5022, 2366, 117, 15186, 23430,
#         117, 126, 1510, 25555, 25102, 2834, 615, 38577, 3188, 117,
#         44582, 807, 117, 1477, 7041, 266, 106, 25161, 32527, 15553,
#         3399, 19285, 117, 3325, 4534, 297, 17634, 1982, 4785, 106,
#         32615, 4656, 117, 26203, 6279, 5728, 14478, 3399, 12604, 33583,
#         15260, 1101, 33558, 4863, 1823, 117, 32012, 6451, 4717, 142,
#         24800, 275, 4297, 42769, 18477, 1670, 4785, 179, 15894, 1741,
#         1779, 1267, 3054, 117, 352, 1741, 2593, 3855, 179],
#        [2269, 4714, 3915, 168, 106, 118, 118, 6086, 48562, 100,
#         47741, 131, 168, 107, 6152, 49662, 107, 131, 121, 117,
#         107, 36034, 48886, 100, 5968, 49738, 107, 131, 35320, 117,
#         107, 36034, 48886, 100, 36048, 107, 131, 107, 5773, 48692,
#         120, 5781, 48678, 48653, 48605, 48559, 107, 117, 107, 6189,
#         48563, 107, 131, 43477, 117, 107, 36034, 48886, 100, 6195,
#         48759, 48567, 107, 131, 35425, 117, 107, 6149, 107, 131,
#         107, 6165, 49032, 48830, 48574, 107, 117, 107, 162, 48970,
#         107, 131, 107, 35220, 48644, 120, 5749, 48663, 48644, 48681,
#         48607, 107, 117, 107, 163, 48580, 107, 131, 107, 5756,
#         48695, 1510, 25563, 25102, 2834, 22603, 2132, 1456, 1370, 830,
#         269, 15186, 4606, 4625, 7944, 27868, 25960, 30330, 18643, 188,
#         32615, 1080, 15386, 21575, 15553, 189, 137, 42489, 4977, 131,
#         22111, 17676, 117, 627, 5284, 3399, 32615, 1963, 25102, 2834,
#         30796, 615, 15889, 117, 22568, 3399, 1463, 527, 2084, 2613,
#         179, 38440, 8186, 314, 15889, 257, 2335, 221, 15553, 117,
#         16923, 44209, 117, 10548, 1454, 25102, 2834, 3399, 15386, 1935,
#         199, 2604, 2637, 1934, 179, 32615, 23776, 2188, 1866, 2357,
#         297, 3923, 1581, 4534, 10792, 179, 2333, 6992, 26181, 3399,
#         18295, 32011, 12312, 266, 117, 15553, 117, 1874, 14827, 23165,
#         8390, 117, 23165, 25725, 188, 1779, 4863, 106, 23185, 20473,
#         6279, 126, 1510, 38654, 4534, 1982, 4785, 106, 189, 128,
#         2334, 129, 2243, 6842, 127, 3023, 4611, 117, 2228, 259,
#         753, 1466, 534, 5201, 5022, 2366, 117, 15186, 23430, 117,
#         126, 1510, 25555, 25102, 2834, 615, 38577, 3188, 117, 44582,
#         807, 117, 1477, 7041, 266, 106, 25161, 32527, 15553, 3399,
#         19285, 117, 3325, 4534, 297, 17634, 1982, 4785, 106, 32615,
#         4656, 117, 26203, 6279, 5728, 14478, 3399, 12604, 33583, 15260,
#         1101, 33558, 4863, 1823, 117, 32012, 6451, 4717, 142, 24800,
#         275, 4297, 42769, 18477, 1670, 4785, 179, 15894, 1741, 1779,
#         1267, 3054, 117, 352, 1741, 2593, 3855, 179]]
#
# for i in src:
#     rs = tokenizer.decode(i)
#     print(rs)

# # src_txt = [''.join(sent) for sent in src]
# print(list('nine'))
# rs = tokenizer.tokenize()
# print(rs)
# str = [1, 2, 3, 4, 5, 2, 6]
# print(str.remove(1))
# print(str)
#
# l1 = [1, 2, 3]
# l1.remove(1)
# print(l1)
#
# alive_seq = [
#     [1, 2, 4, 1, 2, 5],
#     [2, 3, 2, 3, 2]
# ]
# for i in alive_seq:
#     while 1 in i:
#         i.remove(1)
#     while 2 in i:
#         i.remove(2)
# print(alive_seq)

# ngram_size = 3
# num_hypos, cur_len = alive_seq.shape
#
# if cur_len + 1 >= ngram_size:
#     generated_ngrams = [{} for _ in range(num_hypos)]
#     for idx in range(num_hypos):
#         gen_tokens = alive_seq.numpy()[idx]
#         generated_ngram = generated_ngrams[idx]
#         for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
#             prev_ngram_tuple = tuple(ngram[:-1])
#             generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
#
#     banned_batch_tokens = []
#     for hypo_idx in range(num_hypos):
#         start_idx = cur_len + 1 - ngram_size
#         ngram_idx = tuple(alive_seq.numpy()[hypo_idx][start_idx:cur_len])
#         banned_batch_tokens.append(generated_ngrams[hypo_idx].get(ngram_idx, []))
#
#     for i, banned_tokens in enumerate(banned_batch_tokens):
#         if len(banned_tokens) > 0:
#             print(banned_batch_tokens)
#             print(generated_ngrams)  # {(1, 2): [3], (2, 3): [4], (3, 4): [5]}
#             print(banned_tokens)
# curr_scores[i, banned_tokens] = -float("inf")

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# pred = '[unused1] 法 式 浪 漫 和 日 式 。 [unused3] 精 明 试 驾 雷 诺 塔 利 斯 曼 [unused2]'
# pred_str = pred.replace('[unused0]', '').replace('[unused1]', '').replace('[PAD]', '').replace(
#     '[unused2]', '').replace(r' +', ' ').replace(' [unused3] ', '<q>').replace('[unused3]',
#                                                                                '').strip()
# _pred_str = ''
# print(pred_str)
# for sent in pred_str.split('<q>'):
#     can_pred_str = _pred_str + '<q>' + sent.strip()
#     print(can_pred_str)
# print(join('./model/', 'hh.txt'))
# tt = torch.rand(6, 1)
# print(tt)
# tt2 = torch.rand(6, 2)
# print(tt2)
# print(tt + tt2)
# 234
# n = torch.tensor([[[0.3868, 0.0765, 0.1723, 0.8590],
#      [0.1949, 0.5298, 0.8653, 0.1404],
#      [0.2591, 0.4169, 0.5280, 0.9185]],
#
#     [[0.0458, 0.8599, 0.7351, 0.5765],
#      [0.8528, 0.4773, 0.9363, 0.2881],
#      [0.2641, 0.1108, 0.7972, 0.7295]]])
# a = torch.tensor([[[0.3868, 0.0765, 0.1723, 0.8590],
#      [0.1949, 0.5298, 0.8653, 0.1404],
#      [0.2591, 0.4169, 0.5280, 0.9185]],
#
#     [[0.0458, 0.8599, 0.7351, 0.5765],
#      [0.8528, 0.4773, 0.9363, 0.2881],
#      [0.2641, 0.1108, 0.7972, 0.7295]]])
# print(n)
# c = 10
# n_batch, n_sents, n_tokens = n.shape
#
# n = n[:, :, :c // n_sents].contiguous()
#
# n = n.view(n_batch, -1)
# print(n)
# a = a.view(n_batch, -1)
# print(a.shape)
# a = a[:, :6].contiguous()
# print(a)
# n = torch.tensor(n)
# print(n)
# print(n.shape)
#
# n =  n.view(n.size(0), -1)
# print(n)

# MODEL_NAME = 'bert-base-chinese'
# print(f'Loading {MODEL_NAME} Model...')
#
# # 加载模型和tokenizer
# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# model = BertModel.from_pretrained(MODEL_NAME)
#
# # 输入文本并进行tokenizer
# text = ['你好世界!', 'Hello python!']
# inputs = tokenizer(text, return_tensors='pt', padding=True)
# for i in text:
#     print(tokenizer.tokenize(i))
# print('inputs.shape:', inputs['input_ids'].shape)
# print(inputs)
#
# output = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
# pooled_sentence = output.last_hidden_state  # shape is [batch_size, seq_len, hidden_size]
# # pooled_sentence will represent the embeddings for each word in the sentence
# # you need to sum/average the pooled_sentence
# print('pooled_sentence.shape:', pooled_sentence.shape)
# print('-' * 100)
#
# # 池化模型
# pooling_model = MultiHeadedPooling(8, 768)
# context = pooling_model(pooled_sentence, pooled_sentence, mask=inputs['attention_mask'])
# print('-' * 100)
# print('context.shape:', context.shape)
# print(context)
# print('-' * 100)
#
# poolinglayer = TransformerPoolingLayer(768, 8, 2048, 0.1)
# poolinglayer_output = poolinglayer(pooled_sentence, mask=inputs['attention_mask'])
# print('poolinglayer_output.shape', poolinglayer_output.shape)
# print('-'*100)
#
# pooled_sentence = torch.mean(pooled_sentence, dim=1)
# # 得到n_sample*512的句向量
# print('pooled_sentence.shape:', pooled_sentence.shape)
# print(pooled_sentence)

# spend = 10000
# print(r'所花费的时间为：{:.0f}时{:.0f}分{:.0f}秒'.format(spend // 3600, spend % 3600 // 60,
#                                                              spend % 60))
# start = time.time()
# time.sleep(2)
# spend = time.time() - start
# steps = 30000
# print(r'训练{}步所花费的时间为：{:.0f}分{:.0f}秒'.format(steps, spend // 60, spend % 60))
# doc_split = sent_token_split("高质量的网站内容：1、链接跳转。其二、高质量的网站内容：1、链接跳转。", is_short_summary=False)  # 分句
# print(doc_split)
# doc_modified = re.sub(r'其？[0-9一二三四五六七八九十]+[，、.]', "", doc_modified)
# src = [["\u2460", "\u5f00", "\u8f66", "\u65f6", "\u4e5f", "\u4f1a", "\u7761", "\uff0c"],
#        ["\u8d85", "\u957f", "\u7761", "\u7720", "\u4e0d", "\u89e3", "\u56f0", "\uff1b"],
#        ["\u2461", "\u4e0a", "\u8bfe", "\u63d0", "\u95ee", "\u65f6", "\u7ad9", "\u7740", "\u80fd", "\u7761",
#         "\u7740",
#         "\uff0c"],
#        ["\u53d1", "\u4f5c", "\u6027", "\u7761", "\u75c5", "\u591a", "\u53d1", "\u4e8e", "\u9752", "\u5c11",
#         "\u5e74",
#         "\uff1b"], ["\u2462", "\u4e00", "\u7b11", "\u5c31", "\u762b", "\uff0c"],
#        ["\u5178", "\u578b", "\u7761", "\u7720", "\u969c", "\u788d", "\u9762", "\u5bb9", "\u4e0d", "\u80fd",
#         "\u53d7",
#         "\u60c5", "\u7eea", "\u523a", "\u6fc0", "\uff1b"],
#        ["\u2463", "\u7761", "\u7740", "\u4e86", "\u5c31", "\u9192", "\u4e0d", "\u8fc7", "\u6765", "\uff0c"]]
# print(src)
# a = {"title": "#\u5317\u4eac\u7535\u5f71\u8282#\u72ec\u5bb6\u4e13\u8bbf", "content": "\u9646\u5ddd\uff1a\u6211\u7684\u56fd\u7c4d\u6ca1\u597d\u7535\u5f71\u91cd\u8981\u9646\u5ddd\u77e5\u9053\u4e2d\u56fd\u7535\u5f71\u5728\u9ad8\u901f\u53d1\u5c55\u4e4b\u4e0b\u9690\u85cf\u7684\u95ee\u9898\u548c\u5371\u9669\u3002\u4ed6\u8bf4\u6211\u4eec\u4e0d\u8981\u6253\u51fb\u300a\u5c0f\u65f6\u4ee3\u300b\u548c\u300a\u7238\u7238\u53bb\u54ea\u513f\u300b\uff0c\u4f46\u5f53\u4ed6\u7279\u610f\u628a\u8fd9\u4e24\u90e8\u7535\u5f71\u653e\u5230\u53f0\u9762\u4e0a\uff0c\u63a5\u7740\u8bf4\u5e02\u573a\u4e0a\u6709\u592a\u591a\u4eba\u62b1\u7740\u201c\u635e\u4e00\u628a\u201d\u7684\u5fc3\u6001\u65f6\uff0c\u6211\u4eec\u77e5\u9053\u9646\u5ddd\u5bfc\u6f14\u7684\u6001\u5ea6\u5176\u5b9e\u4ece\u672a\u6539\u53d8\u3002"}
# print(a)
# from pyrouge import Rouge155
#
# r = Rouge155("D:\\ProgramData\\GitRepository\\pyrouge-master\\tools\\ROUGE-1.5.5")  # 根据自己rouge放的位置
#
# r.system_dir = 'path/to/system_summaries'
#
# r.model_dir = 'path/to/model_summaries'
#
# r.system_filename_pattern = 'some_name.(\d+).txt'
#
# r.model_filename_pattern = 'some_name.[A-Z].#ID#.txt'
#
# output = r.convert_and_evaluate()

# print(output)

# output_dict = r.output_to_dict(output)

# source = [["\u65b0", "\u5e74", "\u4f0a", "\u59cb", "\uff0c", "\u4e2d", "\u77f3", "\u6cb9", "\u7684", "\u6df7", "\u5408", "\u6240", "\u6709", "\u5236", "\u6539", "\u9769", "\u7387", "\u5148", "\u5728", "\u65b0", "\u7586", "\u7834", "\u9898", "\u3002"], ["\u540c", "\u65f6", "\uff0c", "\u4e0a", "\u8bc1", "\u62a5", "\u8bb0", "\u8005", "\u72ec", "\u5bb6", "\u83b7", "\u6089", "\uff0c", "\u7ee7", "\u53bb", "\u5e74", "\u7684", "\u7535", "\u6539", "\u4e4b", "\u540e", "\uff0c", "\u6cb9", "\u6539", "\u4eca", "\u5e74", "\u5c06", "\u4f1a", "\u201c", "\u5927", "\u52a8", "\u201d", "\uff0c", "\u4e0a", "\u6e38", "\u52d8", "\u63a2", "\u5f00", "\u53d1", "\u548c", "\u4e00", "\u4e9b", "\u5927", "\u4f01", "\u4e1a", "\u7684", "\u5929", "\u7136", "\u6c14", "\u8fdb", "\u53e3", "\u6743", "\u6709", "\u671b", "\u90e8", "\u5206", "\u653e", "\u5f00", "\uff0c", "\u4e24", "\u5927", "\u77f3", "\u6cb9", "\u516c", "\u53f8", "\u7684", "\u6df7", "\u5408", "\u6240", "\u6709", "\u5236", "\u6539", "\u9769", "\u4e5f", "\u5c06", "\u6df1", "\u5316", "\u3002"]]
# tgt = [["\u65b0", "\u4e00", "\u8f6e", "\u201c", "\u6cb9", "\u6539", "\u201d", "\u8feb", "\u8fd1", "\u5929", "\u7136", "\u6c14", "\u8fdb", "\u53e3", "\u6743", "\u6709", "\u671b", "\u90e8", "\u5206", "\u653e", "\u5f00"]]

# src = [["\u745e", "\u5178", "\u662f", "\u4e16", "\u754c", "\u4e0a", "\u4fe1", "\u606f", "\u4e0e", "\u901a", "\u4fe1", "\u6280", "\u672f", "\u53d1", "\u5c55", "\u548c", "\u4f7f", "\u7528", "\u7a0b", "\u5ea6", "\u6700", "\u9ad8", "\u7684", "\u7ecf", "\u6d4e", "\u4f53", "\uff0c", "\u968f", "\u540e", "\u662f", "\u65b0", "\u52a0", "\u5761", "\u3001", "\u82ac", "\u5170", "\u3001", "\u4e39", "\u9ea6", "\u548c", "\u745e", "\u58eb", "\u7b49", "\u3002"], ["\u4e9a", "\u6d32", "\u7ecf", "\u6d4e", "\u4f53", "\u4e2d", "\uff0c", "\u4e2d", "\u56fd", "\u53f0", "\u6e7e", "\u548c", "\u9999", "\u6e2f", "\u5206", "\u522b", "\u6392", "\u540d", "\u7b2c", "1", "1", "\u548c", "\u7b2c", "1", "3", "\u4f4d", "\uff0c", "\u97e9", "\u56fd", "\u548c", "\u65e5", "\u672c", "\u5206", "\u5217", "\u7b2c", "1", "2", "\u4f4d", "\u548c", "1", "8", "\u4f4d", "\u3002"], ["\u6b64", "\u5916", "\uff0c", "\u4e2d", "\u56fd", "\u5185", "\u5730", "\u548c", "\u5370", "\u5ea6", "\u7684", "\u6392", "\u540d", "\u5206", "\u522b", "\u4e3a", "\u7b2c", "5", "1", "\u4f4d", "\u548c", "6", "9", "\u4f4d", "\u3002"]]
# tgt = [["\u6700", "\u65b0", "\u5168", "\u7403", "\u4fe1", "\u606f", "\u6280", "\u672f", "\u6392", "\u540d", "\u4e2d", "\u56fd", "\u6392", "\u540d", "\u7b2c", "5", "1", "\u4f4d"]]
# print('src', len(src), src)
# print('tgt', len(tgt), tgt)
# oracle_ids = combination_selection(src, tgt, 3)
# print(oracle_ids)

# json_file = {
#     "title": "青海首次野外发现濒危大火烈鸟 尚不清楚具体来源",
#     "content": "中新社西宁11月22日电(赵凛松)青海省林业厅野生动植物和自然保护区管理局高级工程师张毓22日向中新社记者确认:“经过中国林业科学院、中科院新疆生态与地理研究所和青海省林业厅的共同认定,"
#                "出现在青海省海西州境内的三只体型较大的鸟为世界极度濒危的红鹳目红鹳科红鹳属的大红鹳。”11月18日,青海省海西州可鲁克湖—托素湖国家级陆生野生动物疫源疫病监测站在野外监测巡护过程中,"
#                "在可鲁克湖西南岸入水口盐沼滩发现三只体型较大的鸟类。张毓说:“此前在该区域从未发现过这种体型的鸟类。”可鲁克湖—托素湖位于青海省柴达木盆地东北部,海拔2800米,"
#                "水域湿地环境内的优势种动物主要是水禽,共有30余种。根据拍摄的照片以及视频,张毓根据动物学体型得出了初步结论,然后会同中国林业科学院和中科院新疆生态与地理研究所的相关专家,"
#                "确认了这三只鸟为红鹳目红鹳科红鹳属的大红鹳。大红鹳也称为大火烈鸟、红鹤等,三只鸟类特征为大红鹳亚成体。根据世界自然保护联盟、世界濒危动物红色名录,该鸟主要分布于非洲、中亚、南亚等区域,"
#                "分布广、种群数量较大,无威胁因子,以往在中国并无分布。但1997年在新疆野外首次发现并确定该鸟在中国境内有分布,为中国鸟类新纪录,"
#                "2012年在四川也发现一只该鸟亚成体。此次野外发现在中国属第三次。“我们现在还无法判断这三只鸟从何而来。不过我个人倾向于是从中亚国家迁徙至此。”张毓强调说,该种鸟国内也有人工饲养,"
#                "因此也有人判断为从动物园逃逸。“我们对这三只鸟进行了详尽的记录,如果明年这个时间还在此地出现这种鸟,那就能肯定是迁徙的鸟类,而不是从动物园里跑出来的。”由于目前可鲁克湖—托素湖已开始结冰,"
#                "鸟类采食困难,不排除三只鸟由于无法获得能量补给而进行远距离迁飞的可能。青海省林业厅野生动物行政主管部门将随时做好野外救护的各项准备工作。 "
# }
# # def _format_to_lines(json_element):  # 格式化每一个摘要对
# #     json_element_split = {'src': sent_token_split(json_element['content']),
# #                           'tgt': sent_token_split(json_element['title'], True)}
# # json_element_formatted = _format_to_lines(format(json_file, 'utf-8'))
# print(sent_token_split(json_file['title'], True))
