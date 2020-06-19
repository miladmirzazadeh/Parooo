import time
import json
import heapq
import re
from sanic import Sanic
from sanic.response import json
from requests_html import AsyncHTMLSession, requests
from requests import urllib3
from random import choice, randint
from loguru import logger
import threading
import time


'''
			http://spys.one
			https://free-proxy-list.net/
			https://www.sslproxies.org/
			https://www.freeproxy.world/
			https://github.com/TheSpeedX/PROXY-List
			https://github.com/chill117/proxy-lists/tree/master/sources
			https://github.com/clarketm/proxy-list/blob/master/proxy-list.txt
			https://github.com/Undercore/ProxyScraper.py/blob/master/SourceCode.py


'''

class ProxyProvider:
    PROTOCOLS = ["https", "http", "socks5"]
    def __init__(self):
        self.ip_q = {}
        self.ip_map = {}
        self.last_ips = {}
        self.last_ips_time = {}
        self.bad_last_ips = {}
        self.bad_last_ips_time = {}
        for p in self.PROTOCOLS:
            self.ip_q[p] = []
            self.ip_map[p] = {}
            self.last_ips[p] = []
            self.last_ips_time[p] = []
            self.bad_last_ips[p] = []
            self.bad_last_ips_time[p] = []
        self.readed_file = False
        self.last_pubproxy_time = time.time()
        self.last_scylla_time = time.time()
        self.last_save_time = time.time()
        self.last_proxy_url_time = time.time()
        self.is_firt_time = True
        self.update_thread = threading.Thread(target=self.thread_update)
        self.update_thread.start()

    def add_ip_with_priority(self, protocol, ip, priority):
        if (len(self.ip_map[protocol]) > 0 and ip in self.ip_map[protocol]) or len(ip) < 10:
            return
        logger.debug(f"add ip: {ip}-{priority}")
        self.ip_map[protocol][ip] = priority
        heapq.heappush(self.ip_q[protocol], (priority, ip))
    
    def add_ip(self, protocol, ip):
        if (len(self.ip_map[protocol]) > 0 and ip in self.ip_map[protocol]) or len(ip) < 10:
            return
        logger.debug(f"add ip: {ip}")
        self.ip_map[protocol][ip] = 2
        heapq.heappush(self.ip_q[protocol], (2, ip))

    def bad_ip(self, protocol, ip):
        if ip in self.ip_map[protocol]:
            for i in range(len(self.ip_q[protocol])):
                if self.ip_q[protocol][i][1] == ip:
                    self.ip_q[protocol][i] = (min(self.ip_q[protocol][i][0] + 2, 6), self.ip_q[protocol][i][1])
            v = self.ip_map[protocol][ip]
            self.ip_map[protocol][ip] = min(v + 2, 6)
            heapq.heapify(self.ip_q[protocol])
    
#     def del_ip(self, ip):
#     def del_bads(self):
    def del_priority(self, protocol, priority):
        nodes = []
        try:
            while len(self.ip_q[protocol]) > 0:
                p = heapq.heappop(self.ip_q[protocol])
                if p[0] != priority:
                    nodes.append(p)
                else:
                    del self.ip_map[protocol][p[1]]
        except:
            logger.exception("WTF in del_proirity")
        for p in nodes:
            self.ip_q[protocol].append(p)
        heapq.heapify(self.ip_q[protocol])

    def _get_good_proxy(self, protocol, change_priority=True):
        if len(self.ip_q[protocol]) < 10:
            return
        min_val = self.ip_q[protocol][0][0]
        nodes = []
        try:
            while len(self.ip_q[protocol]) > 0 and self.ip_q[protocol][0][0] <= min_val+3:
                nodes.append(heapq.heappop(self.ip_q[protocol]))
            retry = True
            while retry:
                while len(self.last_ips_time[protocol]) > 0 and time.time() - self.last_ips_time[protocol][0] > 30:
                    self.last_ips[protocol].pop(0)
                    self.last_ips_time[protocol].pop(0)
                ind = randint(0, len(nodes) - 1)
                if len(self.last_ips[protocol]) + 5 >= len(nodes):
                    time.sleep(3)
                else:
                    retry = False
            while nodes[ind][1] in self.last_ips[protocol]:
                ind = randint(0, len(nodes) - 1)
            if change_priority:
                nodes[ind] = (max(nodes[ind][0] - 1, 0), nodes[ind][1])
            else:
                nodes[ind] = (nodes[ind][0], nodes[ind][1])
        except:
            logger.exception("error in get_good_proxy")
        for i in range(len(nodes)):
            heapq.heappush(self.ip_q[protocol], nodes[i])
        self.ip_map[protocol][nodes[ind][1]] = nodes[ind][0]
        self.last_ips[protocol].append(nodes[ind][1])
        self.last_ips_time[protocol].append(time.time())
        return nodes[ind][1]
    
    def get_bad_proxy(self, protocol):
        if len(self.ip_q[protocol]) < 10:
            return
        min_val = self.ip_q[protocol][0][0]
        nodes = []
        min_val_ind = 0
        try:
            while len(self.ip_q[protocol]) > 0 and self.ip_q[protocol][0][0] < 8:
                if self.ip_q[protocol][0][0] <= min_val+3:
                    min_val_ind = min_val_ind + 1
                nodes.append(heapq.heappop(self.ip_q[protocol]))
            if len(nodes) - min_val_ind < 5:
                for i in range(len(nodes)):
                    self.ip_q[protocol].append(nodes[i])
                heapq.heapify(self.ip_q[protocol])
                return
            retry = True
            while retry:
                while len(self.bad_last_ips_time[protocol]) > 0 and time.time() - self.bad_last_ips_time[protocol][0] > 30:
                    self.bad_last_ips[protocol].pop(0)
                    self.bad_last_ips_time[protocol].pop(0)
                ind = randint(min_val_ind, len(nodes) - 1)
                if len(self.bad_last_ips[protocol]) + 5 >= len(nodes):
                    time.sleep(3)
                else:
                    retry = False
            while nodes[ind][1] in self.bad_last_ips[protocol]:
                ind = randint(min_val_ind, len(nodes) - 1)
            nodes[ind] = (max(nodes[ind][0] - 1, 0), nodes[ind][1])
        except:
            logger.exception("WTF in get_bad func")
        for i in range(len(nodes)):
            self.ip_q[protocol].append(nodes[i])
        heapq.heapify(self.ip_q[protocol])
        self.ip_map[protocol][nodes[ind][1]] = nodes[ind][0]
        self.bad_last_ips[protocol].append(nodes[ind][1])
        self.bad_last_ips_time[protocol].append(time.time())
        return nodes[ind][1]

    def change_priority(self, protocol, src_p, dest_p):
        logger.debug(f"change priority {src_p}   {dest_p}")
        for i in range(len(self.ip_q[protocol])):
            if self.ip_q[protocol][i][0] == src_p:
                self.ip_map[protocol][self.ip_q[protocol][i][1]] = dest_p
                self.ip_q[protocol][i] = (dest_p, self.ip_q[protocol][i][1])
        heapq.heapify(self.ip_q[protocol])
        
    def get_stat(self, protocol):
        if len(self.ip_q[protocol]) > 0:
            min_val = self.ip_q[protocol][0][0]
        else:
            min_val = 6
        cnt = 0
        bins = buckets = [0] * 16
        for p,ip in self.ip_q[protocol]:
            if p <= min_val+3:
                cnt += 1
            bins[p] += 1
        return {"prot": protocol, "# ip_q": len(self.ip_q[protocol]), "# last_ips": len(self.last_ips[protocol]), "# goods": cnt, " bad_last": len(self.bad_last_ips[protocol]), "ips: ": bins}
    
    def save_to_file(self, file_name='./proxies0'):
        for protocol in self.PROTOCOLS:
            with open(f"{file_name}_{protocol}.txt", 'w') as f:
                for p in self.ip_q[protocol]:
                    f.write(f"{p[0]},{p[1]}\n")
    
    def _add_file_proxies(self):
        try:
            self.readed_file = True
            for protocol in self.PROTOCOLS:
                f = open(f"./proxies_{protocol}.txt", "r")
                l = f.readline()
                ps = []
                while l:
                    ps.append(l.replace("\n", "").split(","))
                    l = f.readline()
                for p in ps:
                    self.add_ip_with_priority(protocol, p[1], int(p[0]))
        except:
            logger.exception("couldnt read proxy file")
    
    def _add_pubproxy(self):
        try:
            if len(self.ip_q["https"]) > 10:
                proxies = {"https": self._get_good_proxy("https", change_priority=False)}
            else:
                proxies = None
            for protocol in self.PROTOCOLS:
                self.add_ip(protocol, requests.get("https://pubproxy.com/api/proxy?type=http&speed=15&https=true", timeout=15, proxies=proxies).json()["data"][0]["ipPort"])
        except:
            logger.exception("pubproxy failed")
        pass
    
    def _add_scylla(self):
        try:
            all_p = list(requests.get("http://localhost:8899/api/v1/proxies?https=true&limit=10000", timeout=10).json()["proxies"])
            for p in all_p:
                self.add_ip("https", f'{p["ip"]}:{p["port"]}')
            all_p = list(requests.get("http://localhost:8899/api/v1/proxies?https=false&limit=10000", timeout=10).json()["proxies"])
            for p in all_p:
                self.add_ip("http", f'{p["ip"]}:{p["port"]}')
        except:
            logger.exception("scylla failed")
            pass

    def _add_proxy_url(self):
        urls = []
        with open("proxy_url.txt", "r") as f:
            for line in f.readlines():
                urls.append(line[:-1])
        logger.info(f"number of proxy url {len(urls)}")
        for url in urls:
            logger.debug(f"update {url}")
            try:
                if len(self.ip_q["https"]) > 10:
                    proxies = {"https": self._get_good_proxy("https", change_priority=False)}
                else:
                    proxies = None
                resp = requests.get(f"https://{url}", timeout=15, proxies=proxies)
                regs = re.findall(r'((?:\d{1,3}\.){3}\d{1,3}):(\d+)', resp.text)
                for protocol in self.PROTOCOLS:
                    for reg in regs:
                        self.add_ip(protocol, reg[0] + ":" + reg[1])
                logger.debug(f"done url {url}")
            except:
                try:
                    if len(self.ip_q["http"]) > 10:
                        proxies = {"http": self._get_good_proxy("http", change_priority=False)}
                    else:
                        proxies = None
                    resp = requests.get(f"http://{url}", timeout=15, proxies=proxies)
                    regs = re.findall(r'((?:\d{1,3}\.){3}\d{1,3}):(\d+)', resp.text)
                    for protocol in self.PROTOCOLS:
                        for reg in regs:
                            self.add_ip(protocol, reg[0] + ":" + reg[1])
                    logger.debug(f"done url {url}")
                except:
                    try:
                        resp = requests.get(url, timeout=15)
                        regs = re.findall(r'((?:\d{1,3}\.){3}\d{1,3}):(\d+)', resp.text)
                        for protocol in self.PROTOCOLS:
                            for reg in regs:
                                self.add_ip(protocol, reg[0] + ":" + reg[1])
                        logger.debug(f"done url {url}")
                    except:
                        logger.info(f"bad url {url}")
    
    def thread_update(self):
        time.sleep(10)
        logger.info("update thread start")
        while True:
            logger.info("updating")
            self.update_data()
            time.sleep(30)
    
    def update_data(self):
        if time.time() - self.last_save_time > 300:
                self.save_to_file("proxies-backup")
                logger.info("save backup")
                self.last_save_time = time.time()

        if not self.readed_file:
            self._add_file_proxies()
            logger.info("updated from file")

        if time.time() - self.last_scylla_time > 60 or self.is_firt_time:
            self._add_scylla()
            logger.info("updated scylla")
            self.last_scylla_time = time.time()

        if time.time() - self.last_pubproxy_time > 30 or self.is_firt_time:
            self._add_pubproxy()
            logger.info("updated pub")
            self.last_pubproxy_time = time.time()

        if time.time() - self.last_proxy_url_time > 3600 or self.is_firt_time:
            self._add_proxy_url()
            logger.info("updated url")
            self.last_proxy_url_time = time.time()

        self.is_firt_time = False
            
    def get_proxy(self, protocol):
        return self._get_good_proxy(protocol)

app = Sanic()
proxy_provider = ProxyProvider()

@app.route('/get_proxy/<protocol>')
async def get(request, protocol):
    try:
        return json({"proxy": proxy_provider.get_proxy(protocol)})
    except:
        logger.exception("WTF in get_proxy")
        return json({"proxy": None})

@app.route('/get_bad_proxy/<protocol>')
async def get_bad_proxy(request, protocol):
    try:
        return json({"proxy": proxy_provider.get_bad_proxy(protocol)})
    except:
        logger.exception("WTF in get_bad_proxy")
        return json({"proxy": None})
    
@app.route('/bad_ip/<protocol>/<ipport>')
async def bad_ip(request, protocol, ipport):
    try:
        proxy_provider.bad_ip(protocol, ipport)
        return json({"result": "done"})
    except:
        logger.exception("WTF in bad_ip")
        return json({"result": None})

@app.route('/add_ip/<ipport>')
async def add_ip(request, ipport):
    try:
        for protocol in ProxyProvider.PROTOCOLS:
            proxy_provider.add_ip(protocol, ipport)
        return json({"result": "done"})
    except:
        logger.exception("WTF in add_ip")
        return json({"result": None})

@app.route('/save')
async def save(request):
    try:
        proxy_provider.save_to_file()
        return json({"result": "done"})
    except:
        logger.exception("WTF in save")
        return json({"result": None})

@app.route('/stat/<protocol>')
async def stat(request, protocol):
    try:
        return json(proxy_provider.get_stat(protocol))
    except:
        logger.exception("WTF in stat")
        return json({"result": None})

@app.route('/chng_p/<protocol>/<src_p>/<dest_p>')
async def chng_p(request, protocol, src_p, dest_p):
    try:
        proxy_provider.change_priority(protocol, int(src_p), int(dest_p))
        return json({"result": "done"})
    except:
        logger.exception("WTF in chng_p")
        return json({"result": None})

@app.route('/del_p/<protocol>/<priority>')
async def del_p(request, protocol, priority):
    try:
        proxy_provider.del_priority(protocol, int(priority))
        return json({"result": "done"})
    except:
        logger.exception("WTF in del_p")
        return json({"result": None})

if __name__ == '__main__':
    def setup_logger(file_name):
            logger.remove()
            logger.add(f"./logs/{file_name}-debug.log", format="{time} {level} {message}", level="DEBUG", enqueue=True)
            logger.add(f"./logs/{file_name}-info.log", format="{time} {level} {message}", level="INFO", enqueue=True, backtrace=True)
            logger.add(f"./logs/{file_name}-error.log", format="{time} {level} {message}", level="ERROR", enqueue=True, backtrace=True, diagnose=True)

    setup_logger("proxy_server")
    
    app.run(host='0.0.0.0', port=8008)