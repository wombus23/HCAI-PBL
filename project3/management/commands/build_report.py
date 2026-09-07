"""Rebuild the project 3 PDF report from the cached results."""

from django.core.management.base import BaseCommand

from project3.report import build


class Command(BaseCommand):
    help = "Build the project 3 PDF report (needs reportlab installed)."

    def handle(self, *args, **options):
        try:
            path = build()
        except ImportError:
            self.stderr.write("This command needs reportlab: pip install reportlab")
            return
        self.stdout.write(self.style.SUCCESS("Report written to %s" % path))
