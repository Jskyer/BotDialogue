import json
import asyncio
from flask import Flask, jsonify, request, redirect, url_for, Response, stream_with_context

import langchain_init

# global initialize, 在所有请求到来之前会执行一次
app = Flask(__name__)  # 实例化
print('>>>>>>>>>>>>> flask_before_first_request_config start')
langchain_init.rag_prepare()
print('>>>>>>>>>>>>> flask_before_first_request_config over')


@app.route("/app", methods=['POST'])
def chat_normal():
    data = request.get_json()
    msg = data.get('msg')
    print(msg)

    if not msg:
        print("param error")
        return jsonify({'code': -100, 'msg': 'param error', 'data': None})

    res = langchain_init.response_from_llm(msg)
    return jsonify({
        'code': 200,
        'msg': 'success',
        'data': res
    })
    # return "hello flask"


@app.route("/app/batch", methods=['POST'])
def chat_llm_batch():
    data = request.get_json()
    msg_list = data.get('msg_list')

    if not msg_list or len(msg_list) == 0:
        print("param error")
        return jsonify({'code': -100, 'msg': 'param error', 'data': None})

    res = langchain_init.response_from_llm_batch(msg_list)
    return jsonify({
        'code': 200,
        'msg': 'success',
        'data': res
    })




@app.route("/app/stream", methods=['POST'])
def chat_llm_stream():
    data = request.get_json()
    msg = data.get('msg')
    print(msg)

    if not msg:
        print("param error")
        return jsonify({'code': -100, 'msg': 'param error', 'data': None})

    def stream_func():
        for s in langchain_init.response_from_llm_stream(question=msg):
            yield json.dumps({'event': "message",
                              'id': 1,
                             'data': s}).encode('utf-8')


    return Response(stream_with_context(stream_func()), mimetype="text/event-stream")



@app.route("/app/astream", methods=['POST'])
async def chat_llm_astream():
    data = request.get_json()
    msg = data.get('msg')
    print(msg)

    if not msg:
        print("param error")
        return jsonify({'code': -100, 'msg': 'param error', 'data': None})


    # def stream_func():
    #     for s in langchain_init.response_from_llm_stream(question=msg):
    #         yield json.dumps({'event': "message",
    #                           'id': 1,
    #                           'data': s}).encode('utf-8')
    #
    # return Response(stream_with_context(stream_func()), mimetype="text/event-stream")

    res = await langchain_init.response_from_llm_astream(question=msg)
    print(res)
    # print(type(res))

    return jsonify({
        'code': 200,
        'msg': 'success',
        'data': res
    })


# 定义拦截器函数
def wapper(func):
    print("wapper")
    print(langchain_init.retriever)

    def inner():
        print("执行了装饰器")
        return func()

    return inner


# 定义拦截器函数
def interceptor(func):
    print("interceptor")
    langchain_init.rag_prepare()

    def wrapper():
        print('wrapper')

        print('pre request over')
        return func()

    return wrapper



@app.route("/app/rag", methods=['POST'])
# @interceptor
def chat_llm_rag():
    data = request.get_json()
    msg = data.get('msg')
    print(msg)

    if not msg:
        print("param error")
        return jsonify({'code': -100, 'msg': 'param error', 'data': None})

    # return jsonify({
    #     'code': 200,
    #     'msg': 'success'
    # })
    res = langchain_init.response_from_llm_rag(msg)
    return jsonify({
        'code': 200,
        'msg': 'success',
        'data': res
    })




@app.route("/app/rag_stream", methods=['POST'])
# @wapper
def chat_llm_rag_stream():
    data = request.get_json()
    msg = data.get('msg')
    print(msg)

    if not msg:
        print("param error")
        return jsonify({'code': -100, 'msg': 'param error', 'data': None})

    # def stream_func():
    #     for s in langchain_init.response_from_llm_rag_stream(msg):
    #         yield json.dumps({
    #             'event': "message",
    #             'id': 1,
    #             'data': s
    #         }).encode('utf-8')

    # return jsonify({
    #     'code': 200,
    #     'msg': 'success'
    # })
    return Response(stream_with_context(langchain_init.response_from_llm_rag_stream(msg)), mimetype="text/event-stream")
    # return Response(stream_with_context(stream_func()), mimetype="text/event-stream")





if __name__ == '__main__':
    # 外部可访问：0.0.0.0
    app.run("0.0.0.0", 5050, debug=False)

    # 本地访问：127.0.0.1
    # app.run("127.0.0.1", 5050, debug=False)