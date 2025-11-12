# ---- Stage 1: Create Conda Environment ----
FROM condaforge/mambaforge AS builder
WORKDIR /usr/app

COPY environment.yml .
RUN mamba env create -f environment.yml && \
    mamba clean --all --yes

# ---- Stage 2: Build Cython + Run Streamlit ----
FROM condaforge/mambaforge

# Copy conda environment from builder
COPY --from=builder /opt/conda/envs/dashboard /opt/conda/envs/dashboard
ENV PATH="/opt/conda/envs/dashboard/bin:/opt/conda/bin:$PATH"

# Activate the environment
ENV PATH="/opt/conda/envs/dashboard/bin:$PATH"
ENV CONDA_DEFAULT_ENV=dashboard

WORKDIR /usr

# Copy application source
COPY src/ src/
COPY setup.py . 
COPY dashboard.py .
COPY utils/ utils/


# # Clean up old compiled files
RUN mv src/scripts/*.py /usr/src
RUN rm src/*.so
RUN rm -r src/scripts

# # Build Cython files
RUN python setup.py build_ext --inplace
RUN rm src/*.c src/*.py

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]