# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.


class Maude(models.Model):
    mdr_report_key = models.BigIntegerField()
    device_problem = models.CharField(max_length=191)
    brand_name = models.TextField(blank=True, null=True)
    manufacturer_name = models.TextField(blank=True, null=True)
    device_name = models.TextField(blank=True, null=True)

    class Meta:
        unique_together = ('mdr_report_key', 'device_problem',)
