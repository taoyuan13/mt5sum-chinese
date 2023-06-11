import pymysql
from kafka import KafkaConsumer, TopicPartition
from nameko.standalone.rpc import ClusterRpcProxy

# MQ配置
config_mq = {'AMQP_URI': "amqp://guest:guest@127.0.0.1"}

db = pymysql.connect(host='localhost', user='root', password='root', database='qqnews', charset='utf8')


def consume_news(topic='qq_news', partition=0):
    consumer = KafkaConsumer(
        bootstrap_servers=['localhost:9092'],
        group_id='news',
        auto_offset_reset='latest'
    )
    topic_partition = TopicPartition(topic, partition)
    consumer.assign([topic_partition])
    consumer.seek_to_end(topic_partition)
    for message in consumer:
        if len(message.value) != 0:
            message_key = message.key.decode()
            message_value = message.value.decode()
            if message_value == '<kill>':
                break
            message_value = [v.split('-.-') for v in message_value.split('=.=')]
            print('message_key:', message_key, '\tpartition:', partition)
            news = [v[-1] for v in message_value]

            with ClusterRpcProxy(config_mq) as rpc:
                datasets = rpc.data_service.preprocess(news, message_key)
                if partition == 0:
                    summaries = rpc.summary_service.ext_summarize(datasets)
                else:
                    summaries = rpc.summary_service.abs_summarize(datasets)
            print('summaries:', partition, summaries)
            for i, v in enumerate(message_value):
                insert_mysql(v[0], v[1], message_key, v[2], summaries[i], v[3])
                # insert_mysql(b[0], b[1], b[2], summaries[i], b[3])
        else:
            print(message.key, message.value)
    db.close()
    print('消费者{}结束了！'.format(partition))


# insertarr = []
# for message in balanced_consumer:
#     print(message)
#     if message is not None:
#         # print(message.offset, message.value, type(message.value), str(message.value, encoding="utf8"))
#         # 将接受到的数据转换成executemany能接受的数据格式
#         arrs = str(message.value, encoding="utf8").split('=.=')
#         for arr in arrs:
#             a = arr.split('-.-')
#             insertarr.append(a)
# try:
#
#     cur = conn.cursor()
#     sql = "INSERT INTO 数据库.表名(字段1,字段2,字段3,字段4,字段5) VALUES(%s,%s,%s,%s,%s)"
#     print(insertarr)
#     cur.executemany(sql, insertarr)
#     conn.commit()
#     insertarr = []
#     conn.close()
# except Exception as e:
#     print("插入错误：%s" % e)
#     insertarr = []
#     conn.close()


def insert_mysql(id, url, task, title, summary, content):
    db.ping(reconnect=True)
    cursor = db.cursor()
    sql = "INSERT INTO news(id, url, task, title, summary, content) VALUES ('%s','%s','%s','%s','%s','%s')" % (
        id, url, task, title, summary, content)
    try:
        cursor.execute(sql)
        db.commit()
        print('插入成功-------------------------------------')
    except Exception as e:
        db.rollback()
        print('插入失败***********************************%s' % e)
