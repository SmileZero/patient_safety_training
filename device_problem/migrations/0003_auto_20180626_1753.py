# -*- coding: utf-8 -*-
# Generated by Django 1.11.13 on 2018-06-26 17:53
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('device_problem', '0002_auto_20180626_1345'),
    ]

    operations = [
        migrations.AlterField(
            model_name='maude',
            name='mdr_report_key',
            field=models.BigIntegerField(max_length=191),
        ),
    ]
