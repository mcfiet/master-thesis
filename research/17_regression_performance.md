## MixUp Regression Training verbessern

Lösung warum MixUp mit dem On-the-Fly Shuffeling nicht gut funktioniert hat:

- Hybrid Lösung zuerst (Am Anfang festgelegte Varianz bei der MixUp und dann mit den Epochen wenn einigermasen konvertiert mehr random generieren)
- Learning Rate Scheduler versuchen (Auf ab auf ab)

- testen auf lebenshilfe dataset
