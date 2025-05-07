README
======

This directory contains the files necessary for running a Datasette-based web UI
for the MultiCAT database.

If you want to run the UI locally, you can do the following (assuming you have
[Docker](https://www.docker.com/) installed):

1. Create the `datasette-with-plugins` Docker image by running

  ./create_datasette_with_plugins_image

2. Run `docker compose up` from this directory

You can then navigate to `localhost:8001` in your browser to view the UI.


This directory contains the following:

- The `docker-compose.yml` file specifies the base image, port mapping, volumes,
  and entrypoint command for the UI.
- The `metadata.yml` file contains metadata about the tables and columns in YAML
  format.
- The `multicat.db` SQLite3 database which gets mounted onto the container.
- `templates`: Jinja2 templates for pages.
- `static`: Static files (CSS, images, etc.)

