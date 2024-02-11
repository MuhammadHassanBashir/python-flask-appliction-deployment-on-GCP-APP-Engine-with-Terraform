from retriever import CustomGoogleVertexAISearchRetriever
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import logging
import timeit

logger = logging.getLogger()
LOG_FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)

app = Flask(__name__, static_folder="build/static", static_url_path="/")
load_dotenv()

@app.route('/citation', methods=['POST'])
def search():
    try:
        args = request.json
        query = args.get('query')
        structured_approach = args.get('structured_approach')
        st=timeit.default_timer()
        Retriever = CustomGoogleVertexAISearchRetriever(structured_approach=structured_approach)
        retriever_init = timeit.default_timer()-st
        results = Retriever.get_relevant_documents(query, return_dict=True)
        intents=list(Retriever._get_sorted_intents().keys())
        vertex_summary=Retriever._get_most_relevant_summary()
        timestamps=Retriever.timestamps
        timestamps['retriever_init'] = retriever_init
        return jsonify(success=True,intents=intents,results=results,timestamps=timestamps,vertex_summary=vertex_summary), 200
    except Exception as e:
        return jsonify(success=False, errors=[str(e)]), 500


@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'success':True,'message':'hello there'})


@app.route('/', methods=['GET'])
def test2():
    return jsonify({'success':True,'message':'I am root'})
# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def index():
#     return app.send_static_file("index.html")


if __name__ == '__main__':  
   #app.run(host='0.0.0.0', port=5000)
   #app.run()
   #app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
   app.run(host='0.0.0.0', port=8080)
