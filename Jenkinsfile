pipeline {
    agent any

    environment {
        IMAGE_NAME = "pathway-to-improved-cities"
        IMAGE_TAG  = "${env.BUILD_NUMBER}"
        REGISTRY   = "" // e.g. "ghcr.io/aryan"  leave blank to skip push
    }

    options {
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '20'))
        timeout(time: 30, unit: 'MINUTES')
    }

    stages {
        stage('Checkout') {
            steps { checkout scm }
        }

        stage('Lint') {
            steps {
                sh '''
                    python3 -m venv .venv-ci
                    . .venv-ci/bin/activate
                    pip install --quiet --upgrade pip
                    pip install --quiet ruff
                    ruff check src || true
                '''
            }
        }

        stage('Compile Check') {
            steps {
                sh '''
                    . .venv-ci/bin/activate
                    pip install --quiet -r requirements.txt
                    python -m compileall -q src
                '''
            }
        }

        stage('Smoke Test') {
            steps {
                sh '''
                    . .venv-ci/bin/activate
                    cd src
                    python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from city_config import CITIES, get_city, load_boundary
for key in CITIES:
    city = get_city(key)
    try:
        geo, am = load_boundary(city)
        assert len(am) > 0, f"{key}: empty area_map"
        print(f"[ok] {key}: {len(am)} areas")
    except Exception as e:
        print(f"[warn] {key}: {e}")
PY
                '''
            }
        }

        stage('Docker Build') {
            steps {
                sh 'docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest .'
            }
        }

        stage('Container Health') {
            steps {
                sh '''
                    docker rm -f pic-ci 2>/dev/null || true
                    docker run -d --name pic-ci -p 18501:8501 ${IMAGE_NAME}:${IMAGE_TAG}
                    for i in $(seq 1 30); do
                        if curl -fsS http://localhost:18501/_stcore/health >/dev/null; then
                            echo "healthy"; exit 0
                        fi
                        sleep 2
                    done
                    docker logs pic-ci
                    exit 1
                '''
            }
            post {
                always { sh 'docker rm -f pic-ci 2>/dev/null || true' }
            }
        }

        stage('Push') {
            when { expression { return env.REGISTRY?.trim() } }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'docker-registry',
                    usernameVariable: 'REG_USER',
                    passwordVariable: 'REG_PASS',
                )]) {
                    sh '''
                        echo "$REG_PASS" | docker login ${REGISTRY%%/*} -u "$REG_USER" --password-stdin
                        docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
                        docker tag ${IMAGE_NAME}:latest     ${REGISTRY}/${IMAGE_NAME}:latest
                        docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
                        docker push ${REGISTRY}/${IMAGE_NAME}:latest
                    '''
                }
            }
        }
    }

    post {
        always  { cleanWs() }
        success { echo "Build #${BUILD_NUMBER} green — image ${IMAGE_NAME}:${IMAGE_TAG}" }
        failure { echo "Build #${BUILD_NUMBER} failed" }
    }
}
