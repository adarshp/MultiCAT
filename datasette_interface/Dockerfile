# Datasette plus the plugins the MultiCAT UI uses. Bump the tag deliberately;
# 0.65.5 is on Debian bookworm (0.65.1 was on Debian 11, now end of life).
FROM datasetteproject/datasette:0.65.5
RUN pip install --no-cache-dir \
      datasette-render-markdown==2.2.1 \
      datasette-pretty-json==0.3 \
      datasette-copyable==0.3.2
