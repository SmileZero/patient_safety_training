import requests
from bs4 import BeautifulSoup
import re
import pysolr
import traceback
import logging
from django.db import IntegrityError
from django.conf import settings
from django.db import transaction
from .models import Maude

logger = logging.getLogger(__name__)


class DeviceProblemScrapper:
    def __init__(self, fromID, toID):
        self.fromID = fromID
        self.toID = toID
        self.lastID = toID
        self.finalResult = []

    def parse_p(self, id):
        params = {
            'mdrfoi__id': id,
        }
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
        response = requests.get(settings.SOLR_DNS + 'maude/select?fl=mdr_report_key,%20device.brand_name,' +
                                '%20device.manufacturer_d_name,%20device.generic_name&' +
                                'fq=mdr_report_key:[' + self.fromID + '%20TO%20' + self.toID +
                                ']&q=*:*&rows=100000000&wt=json')
        solr = pysolr.Solr(settings.SOLR_DNS + 'deviceProblems/', timeout=1000000)
        response = response.json()
        logger.info(response['response']['numFound'], "documents found.")

        try:
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
                    try:
                        if par is not None:
                            if len(par) > 1:
                                print(par)
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
                                    Maude.objects.create(mdr_report_key=par[0],
                                                         device_problem=prob,
                                                         brand_name=brand_name,
                                                         manufacturer_name=manufacturer_d_name,
                                                         device_name=generic_name)
                                    dpDictList.append(DPDictionary)
                                lastID = id[0]
                    except IntegrityError:
                        pass
                    except Exception as e:
                        logger.error(e.message())
                        logger.error(traceback.format_exc())
                        logger.error("error stage 1")
                        return
                    logger.info("dumping")
                    solr.add(dpDictList)
                    logger.info("write complete")
        except Exception as e:
            logger.error(e.message())
            logger.error(traceback.format_exc())
            logger.error("error stage 2")
            return

        logger.info("--- %s last record ---" % (lastID))
