# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import JsonResponse
from .training_helper import TrainingHelper
from .models import Maude
import os.path
from django.conf import settings
from django.db.models import Max
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def training(request, end_report_key):
    if request.method == 'POST':
        if os.path.isfile(settings.TRAINING_LOG_FILE):
            return JsonResponse({"status": "already started"})

        start_report_key = Maude.objects.all().aggregate(Max('mdr_report_key')).get('mdr_report_key__max')
        if not start_report_key:
            start_report_key = 0

        TrainingHelper(start_report_key, end_report_key).start()
        return JsonResponse({"status": "ok"})


def status(request):
    if not os.path.isfile(settings.TRAINING_LOG_FILE):
        return JsonResponse({"status": "ready"})

    log_file = open(settings.TRAINING_LOG_FILE, 'r')
    log_content = log_file.read()
    log_file.close()
    return JsonResponse({"status": log_content})
