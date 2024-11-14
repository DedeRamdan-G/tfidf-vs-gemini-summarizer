# myapp/models.py
from django.db import models

class Text(models.Model):
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'myapp_text'

class Summary(models.Model):
    TFIDF = 'TF-IDF'
    AI = 'AI'
    METHOD_CHOICES = [
        (TFIDF, 'TF-IDF'),
        (AI, 'AI'),
    ]

    text = models.ForeignKey(Text, on_delete=models.CASCADE)
    method = models.CharField(max_length=6, choices=METHOD_CHOICES)  
    summary = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'myapp_summaries'

class RougeScore(models.Model):
    summary = models.ForeignKey(Summary, on_delete=models.CASCADE)
    rouge1 = models.FloatField()
    rouge2 = models.FloatField()
    rougeL = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'myapp_rougescores'

class Comparison(models.Model):
    text = models.ForeignKey(Text, on_delete=models.CASCADE)
    tfidf_summary = models.ForeignKey(Summary, related_name='tfidf_summary', on_delete=models.CASCADE)
    ai_summary = models.ForeignKey(Summary, related_name='ai_summary', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'myapp_comparisons'
