import requests
import time
from loguru import logger

class ProxyProvider:
    
    def __init__(self, ipport="0.0.0.0:8008"):
        self.ipport=ipport
    
    def get_bad_proxy(self, protocol):
        while True:
            try:
                p = requests.get(f"http://{self.ipport}/get_bad_proxy/{protocol}").json()["proxy"]
                return p
            except:
                logger.debug("cant get from proxy")
                time.sleep(0.5)
    
    def get_proxy(self, protocol):
        while True:
            try:
                p = requests.get(f"http://{self.ipport}/get_proxy/{protocol}").json()["proxy"]
                return p
            except:
                logger.debug("cant get from proxy")
                time.sleep(0.5)
    
    def bad_ip(self, protocol, ip):
        while True:
            try:
                requests.get(f"http://{self.ipport}/bad_ip/{protocol}/{ip}")
                return
            except:
                logger.debug("cant send bad ip")
                time.sleep(0.5)
    
    def get_link(self, url, protocol=None, retry=-1):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Cafari/537.36'}
        if retry < 0:
            retry = -10
        r = None
        if not protocol:
            protocol = url.split("://")[0]
        
        while retry == -10 or retry >= 0:
            proxies = {protocol: self.get_proxy(protocol)}
            try:
                r = requests.get(f"{protocol}://google.com", headers=headers, proxies=proxies, timeout=10)
                if "google.com" in r.text:
                    r = requests.get(url, headers=headers, proxies=proxies, timeout=10)
                    if r.status_code != 200:
                        self.bad_ip(protocol, proxies[protocol])
                        logger.debug("retry getting")
                    else:
                        return r
                else:
                    self.bad_ip(protocol, proxies[protocol])
                    logger.debug("retry getting")
            except:
                self.bad_ip(protocol, proxies[protocol])
                logger.debug("retry getting")
            retry = max(-10, retry - 1)
        return r
    
    def get_proxy_for_link(self, url, protocol=None, retry=-1):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Cafari/537.36'}
        if retry < 0:
            retry = -10
        r = None
        if not protocol:
            protocol = url.split("://")[0]
        
        while retry == -10 or retry >= 0:
            proxies = {protocol: self.get_proxy(protocol)}
            try:
                r = requests.get(f"{protocol}://google.com", headers=headers, proxies=proxies, timeout=10)
                if "google.com" in r.text:
                    r = requests.get(url, headers=headers, proxies=proxies, timeout=10)
                    if r.status_code != 200:
                        self.bad_ip(protocol, proxies[protocol])
                        logger.debug("retry getting")
                    else:
                        return proxies[protocol]
                else:
                    self.bad_ip(protocol, proxies[protocol])
                    logger.debug("retry getting")
            except:
                self.bad_ip(protocol, proxies[protocol])
                logger.debug("retry getting")
            retry = max(-10, retry - 1)
        return None
        
        