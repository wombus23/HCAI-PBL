"""Rebuild the project 4 PDF report."""

from django.core.management.base import BaseCommand

from project4.report import build


class Command(BaseCommand):
    help = "Build the project 4 PDF report (needs reportlab installed)."

    def handle(self, *args, **options):
        try:
            path = build()
        except ImportError:
            self.stderr.write("This command needs reportlab: pip install reportlab")
            return
        self.stdout.write(self.style.SUCCESS("Report written to %s" % path))
