import json
import redis
import argparse
import random

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qname', type=str, help="redis queue", default="posec")
    parser.add_argument('--host', type=str, help="redis host", default="paxos")
    parser.add_argument('--port', type=int, help="redis port", default=8888)
    args = parser.parse_args()
    r = redis.StrictRedis(host=args.host, port=args.port)

    r.delete(args.qname)
    r.delete(args.qname+'_PROCESSING')

    jobs = []
    for game in ['GSP', 'GFP']:
        for n in reversed(range(2,11,2)):
            for seed in range(1,11):
                job = {'game': game, 'n': n, 'seed': seed, 'bbsi_level': 1}
                jobs.append(job)

    for game in ['GSP_bad', 'GFP_bad']:
        for n in range(2,11,2):
           for seed in range(1,11):
                job = {'game': game, 'n': n, 'seed': seed, 'bbsi_level': 1}
                jobs.append(job)

    for n in range(2,11,2):
        job = {'game': 'vote_bad', 'n': n, 'seed': seed, 'bbsi_level': 1}
        jobs.append(job)

    for n in range(2,11,2):
        job = {'game': 'vote', 'n': n, 'seed': 1, 'bbsi_level': 1}
        jobs.append(job)

    random.shuffle(jobs)

    for job in jobs:
        r.rpush(args.qname, json.dumps(job))
    print r.llen(args.qname)
