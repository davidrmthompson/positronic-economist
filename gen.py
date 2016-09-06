import json
import redis
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qname', type=str, help="redis queue", default="posec")
    parser.add_argument('--host', type=str, help="redis host", default="paxos")
    parser.add_argument('--port', type=int, help="redis port", default=8888)
    args = parser.parse_args()
    r = redis.StrictRedis(host=args.host, port=args.port)

    r.delete(args.qname)
    for game in ['GSP', 'GFP']:
	    for n in reversed(range(2,13)):
	    	for seed in range(1,51):
	    		job = {'game': game, 'n': n, 'seed': seed}
	    		r.rpush(args.qname, json.dumps(job))
    print r.llen(args.qname)
