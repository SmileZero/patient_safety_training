# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.


class Maude(models.Model):
    mdr_report_key = models.CharField(max_length=191)
    device_problem = models.CharField(max_length=191)
    brand_name = models.TextField(blank=True)
    manufacturer_name = models.TextField(blank=True)
    device_name = models.TextField(blank=True)

    class Meta:
        unique_together = ('mdr_report_key', 'device_problem',)
