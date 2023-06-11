# -*- coding: utf-8 -*-
import json
import subprocess
from multiprocessing import Process, Lock

import flask
from flask import Flask, request, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from nameko.standalone.rpc import ClusterRpcProxy

from prepro.tokenizer import T5PegasusTokenizer
from service.kfka_consumer import consume_news


def get_absolute_address():
    return '127.0.0.1'


app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_absolute_address,  # 记录所有访问者的访问次数
    # key_func=get_remote_address,  # 根据访问者的IP记录访问次数
    # default_limits=["200 per day", "50 per hour"]  # 默认限制，一天最多访问200次，一小时最多访问50次
)
config_mq = {'AMQP_URI': "amqp://guest:guest@127.0.0.1"}  # MQ配置
tokenizer = T5PegasusTokenizer.from_pretrained('../t5_pegasus_chinese/', do_lower_case=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analysis', methods=['GET'])
# @limiter.limit("1 per hour")  # 自定义访问速率
def analysis():
    print('访问id:', get_remote_address())
    # 开启两个消费者
    p_list = []
    for i in range(2):
        print('开启第{}个消费者。。。'.format(i))
        p = Process(target=consume_news, args=('qq_news', i))
        p.start()
        p_list.append(p)
    # 调用爬虫微服务
    with ClusterRpcProxy(config_mq) as rpc:
        rpc.spider_service.crawl()
    # 返回结果
    [p.join() for p in p_list]
    print('返回结果')
    return '执行完成！'


@app.route('/predict', methods=['POST'])
def predict():
    # try:
    sentence = request.json['input_text']
    num_words = request.json['num_words']
    num_beams = request.json['num_beams']
    no_repeat = request.json['no_repeat']
    # model = request.json['model']
    if sentence != '':
        print('源文本:', sentence)
        with ClusterRpcProxy(config_mq) as rpc:
            if len(sentence) > 1500:
                datasets = rpc.data_service.preprocess([sentence], 'ext')
                output = rpc.summary_service.ext_summarize(datasets, sort=1, score=0.5)[0]
            else:
                datasets = rpc.data_service.preprocess([sentence], 'abs')
                output = rpc.summary_service.abs_summarize(datasets, int(num_beams), int(num_words), int(no_repeat))[0]
        response = {'response': {
            'summary': str(output),
            # 'model': model.lower()
        }}
        return flask.jsonify(response)
    else:
        res = dict({'message': 'Empty input'})
        return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')
    # except Exception as ex:
    #     res = dict({'message': str(ex)})
    #     print(ex)
    #     return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
