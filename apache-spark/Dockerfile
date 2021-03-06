FROM ubuntu:16.04

MAINTAINER Christoph Champ <cchamp@redapt.com>
LABEL version="spark_2.1_hadoop_2.7"

# Install Python
RUN \
  apt-get update && \
  apt-get install -y python python-dev python-pip python-virtualenv && \
  rm -rf /var/lib/apt/lists/*

# Install R
RUN echo "deb http://cran.rstudio.com/bin/linux/ubuntu xenial/" | tee -a /etc/apt/sources.list && \
    gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9 && \
    gpg -a --export E084DAB9 | apt-key add - && \
    apt-get update && \
    apt-get install -y r-base r-base-dev

# Install system tools
RUN \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y build-essential && \
  apt-get install -y software-properties-common && \
  apt-get install -y curl wget unzip vim git htop byobu && \
  rm -rf /var/lib/apt/lists/*

# Install extra packages required by and/or useful for working with Apache
# Spark
RUN pip install --upgrade pip
RUN pip install ipython numpy scipy sklearn pandas matplotlib

# Install Java
ARG JAVA_MAJOR_VERSION=8
RUN \
  echo oracle-java${JAVA_MAJOR_VERSION}-installer shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
  add-apt-repository -y ppa:webupd${JAVA_MAJOR_VERSION}team/java && \
  apt-get update && \
  apt-get install -y oracle-java${JAVA_MAJOR_VERSION}-installer && \
  rm -rf /var/lib/apt/lists/* && \
  rm -rf /var/cache/oracle-jdk${JAVA_MAJOR_VERSION}-installer

# Define commonly used JAVA_HOME variable
ENV JAVA_HOME /usr/lib/jvm/java-${JAVA_MAJOR_VERSION}-oracle

ARG SPARK_VERSION="v2.1.0"

RUN git clone --depth 1 --branch ${SPARK_VERSION} https://github.com/apache/spark.git

WORKDIR spark

ENV R_HOME /usr/lib/R
RUN ./R/install-dev.sh

ENV MAVEN_OPTS "-Xmx2g -XX:ReservedCodeCacheSize=512m"
ARG MAJOR_HADOOP_VERSION="2.7"
RUN ./build/mvn -Pyarn -Psparkr -Pmesos -Phive -Phive-thriftserver \
  -Phadoop-${MAJOR_HADOOP_VERSION} \
  -Dhadoop.version=${MAJOR_HADOOP_VERSION}.0 \
  -DskipTests clean package

ENV SPARK_HOME /spark
ENV PYSPARK_DRIVER_PYTHON ipython
ENV PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.4-src.zip:${PYTHONPATH}"
ENV PATH="/spark/bin:${PATH}"
