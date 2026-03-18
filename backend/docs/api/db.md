# Database & Job Queue

The `db` module handles the persistence layer of the LumiXAI framework. It utilizes a hybrid storage architecture: lightweight metadata (such as execution times and status flags) is managed by an **SQLite database via SQLAlchemy**, while large attribution tensors and heatmaps are serialized as **JSON files on the host filesystem**. This prevents database bloat and ensures high-speed read/write operations for large models.

## ORM Model

::: src.db.Job
    options:
      show_root_heading: false
      show_root_toc_entry: false

## CRUD Operations

::: src.db.create_job
    options:
      show_root_heading: false
      show_root_toc_entry: false

::: src.db.update_job_success
    options:
      show_root_heading: false
      show_root_toc_entry: false

::: src.db.update_job_failed
    options:
      show_root_heading: false
      show_root_toc_entry: false

::: src.db.get_job
    options:
      show_root_heading: false
      show_root_toc_entry: false

::: src.db.get_all_jobs
    options:
      show_root_heading: false
      show_root_toc_entry: false

::: src.db.delete_all_jobs
    options:
      show_root_heading: false
      show_root_toc_entry: false