pipeline {
    agent any

    environment {
        IMAGE_NAME = "pathway-to-improved-cities"
        IMAGE_TAG  = "${env.BUILD_NUMBER}"
        REGISTRY   = "" // set to e.g. "ghcr.io/aryan" to enable push
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

        stage('Conflict Marker Check') {
            steps {
                sh '''
                    set -e
                    if grep -RIn --include="*.py" -E "^(<<<<<<<|=======|>>>>>>>)" src; then
                        echo "ERROR: unresolved merge conflict markers in src/"
                        exit 1
                    fi
                    echo "[ok] no conflict markers"
                '''
            }
        }

        stage('Docker Build') {
            steps {
                sh 'docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -t ${IMAGE_NAME}:latest .'
            }
        }

        stage('Compile Check') {
            steps {
                sh '''
                    docker run --rm \
                      -v "$PWD":/workspace -w /workspace \
                      ${IMAGE_NAME}:${IMAGE_TAG} \
                      python -m compileall -q src
                '''
            }
        }

        stage('Smoke Test') {
            steps {
                sh '''
                    docker run --rm \
                      -w /app/src \
                      ${IMAGE_NAME}:${IMAGE_TAG} \
                      python -c "
import warnings; warnings.filterwarnings('ignore')
from city_config import CITIES, get_city, load_boundary
failed = 0
for key in CITIES:
    city = get_city(key)
    try:
        geo, am = load_boundary(city)
        assert len(am) > 0, 'empty area_map'
        print(f'[ok] {key}: {len(am)} areas')
    except Exception as e:
        print(f'[FAIL] {key}: {e}')
        failed += 1
exit(1 if failed else 0)
"
                '''
            }
        }

        stage('Container Health') {
            steps {
                sh '''
                    docker rm -f pic-ci 2>/dev/null || true
                    docker run -d --name pic-ci -p 18501:8501 ${IMAGE_NAME}:${IMAGE_TAG}
                    ok=0
                    for i in $(seq 1 30); do
                        if docker exec pic-ci curl -fsS http://localhost:8501/_stcore/health >/dev/null 2>&1; then
                            echo "[ok] container healthy"
                            ok=1
                            break
                        fi
                        sleep 2
                    done
                    if [ "$ok" != "1" ]; then
                        echo "[FAIL] container never became healthy"
                        docker logs pic-ci
                        exit 1
                    fi
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
                        set -e
                        REG_HOST=$(echo "$REGISTRY" | cut -d/ -f1)
                        echo "$REG_PASS" | docker login "$REG_HOST" -u "$REG_USER" --password-stdin
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
