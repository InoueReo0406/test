# 設定ファイルの読み込み
source ./config.txt

#実行
echo $DOCKER_VERSION
curl -LO https://download.docker.com/mac/static/stable/x86_64/docker-$DOCKER_VERSION.tgz
ls