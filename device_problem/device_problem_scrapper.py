import requests
from bs4 import BeautifulSoup
import re
import pysolr
import traceback
from django.conf import settings
from django.db import transaction
from .models import Maude


class DeviceProblemScrapper:
    def __init__(self, fromID, toID, status_logger):
        self.fromID = fromID
        self.toID = toID
        self.lastID = toID
        self.finalResult = []
        self.status_logger = status_logger

    def parse_p(self, id):
        params = {'mdrfoi__id': id}
        response = requests.get('https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfmaude/detail.cfm', params=params)
        html = str(response.content)
        soup = BeautifulSoup(html, 'html.parser')
        result = []
        if ("Device Problem" in html) or ("Device Problems" in html):
            result.append(id)
            regex = re.compile(r'[\\r]+[\\n]+[\\t]+')
            for span in soup.find_all('span', {'style': 'font-family: arial; color: ##23238E;'}):
                result.append(regex.sub('', span.text.strip()))
            return result
        else:
            return None

    def get_data(self):
        response = requests.get('http://' + settings.SOLR_DNS + ':8983/solr/maude/select?fl=mdr_report_key,' +
                                'device.brand_name,device.manufacturer_d_name,device.generic_name&' +
                                'fq=mdr_report_key:[' + str(self.fromID) + '%20TO%20' + str(self.toID) +
                                ']&q=*:*&rows=100000000&wt=json')
        solr = pysolr.Solr('http://' + settings.SOLR_DNS + ':8983/solr/deviceProblems/', timeout=1000000)
        response = response.json()
        self.status_logger.write("%d documents found.\n" % response['response']['numFound'])

        try:
            self.status_logger.write("Writing to MySQL...\n")
            with transaction.atomic():
                dpDictList = []
                for doc in response['response']['docs']:
                    id = doc['mdr_report_key']
                    par = self.parse_p(int(id[0]))
                    brand_name = doc['device.brand_name'][0] if 'device.brand_name' in doc else None
                    generic_name = doc['device.generic_name'][0] if 'device.generic_name' in doc else None
                    manufacturer_d_name = None
                    if 'device.manufacturer_d_name' in doc:
                        manufacturer_d_name = doc['device.manufacturer_d_name'][0]

                    if par is not None:
                        if len(par) > 1:
                            for prob in par[1].split(";"):
                                self.finalResult.append([par[0], prob.strip()])
                                DPDictionary = {
                                    "id": str(par[0]) + prob,
                                    "mdr_report_key": par[0],
                                    "device_problem": prob,
                                    "brand_name": brand_name,
                                    "manufacturer_d_name": manufacturer_d_name,
                                    "generic_name": generic_name,
                                    "unique_key": str(par[0]) + prob,
                                }
                                if prob == "Blank screen":
                                    continue
                                Maude.objects.get_or_create(mdr_report_key=par[0],
                                                            device_problem=prob,
                                                            defaults={
                                                                'brand_name': brand_name,
                                                                'manufacturer_name': manufacturer_d_name,
                                                                'device_name': generic_name
                                })
                                dpDictList.append(DPDictionary)
                            self.lastID = id[0]
                self.status_logger.write("Writing to Solr...\n")
                solr.add(dpDictList)
                self.status_logger.write("Write Complete\n")
        except Exception as e:
            self.status_logger.write("[Error] %s...\n" % e.message)
            self.status_logger.write(traceback.format_exc())
