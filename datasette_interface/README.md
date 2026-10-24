README
======

This directory contains the files necessary for running a Datasette-based web UI
for the MultiCAT database.

If you want to run the UI locally, you can do the following (assuming you have
[Docker](https://www.docker.com/) installed):

1. Run `docker compose up --build` from this directory. It builds the
   `datasette-with-plugins` image from the `Dockerfile` the first time.

You can then navigate to `localhost:8001` in your browser to view the UI.


This directory contains the following:

- The `Dockerfile` pins the Datasette image and the plugins installed on it.
- The `docker-compose.yml` file specifies the image, port mapping, volumes,
  and entrypoint command for the UI.
- The `metadata.yml` file contains metadata about the tables and columns in YAML
  format.
- The `multicat.db` SQLite3 database which gets mounted onto the container.
- `templates`: Jinja2 templates for pages.
- `static`: Static files (CSS, images, etc.)

