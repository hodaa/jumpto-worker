"""Run the Celery worker in-process for local debugging.

``--pool=solo`` runs each task inside the single worker process, so VSCode's
debugger can hit breakpoints (prefork would run tasks in child processes the
debugger never attaches to).
"""

from app.tasks.celery_app import celery_app

if __name__ == "__main__":
    celery_app.worker_main(["worker", "--pool=solo", "--loglevel=info"])
