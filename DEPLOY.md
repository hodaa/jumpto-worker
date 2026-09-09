# Deploy JumpTo Worker to a New Server

## 1. Build the chromium-cdp image

```sh
docker build -f scripts/chromium-cdp-Dockerfile -t chromium-cdp scripts/
```

## 2. Run the Chromium container

```sh
docker run -d --name chromium --restart=unless-stopped \
  -p 127.0.0.1:3001:3001 -p 127.0.0.1:9222:9222 \
  --tmpfs /dev/shm \
  -v /etc/jumpto/chromium/profile:/config \
  -v /etc/jumpto:/etc/jumpto \
  -v /etc/localtime:/etc/localtime:ro \
  chromium-cdp
```

## 3. Sign into YouTube (one-time)

```sh
docker exec chromium chromium-browser --no-sandbox
```

Open YouTube, sign in, then close the browser. The session is saved in the profile volume.

## 4. Set up the cookie refresh cron

```sh
crontab -e
```

Add this line (runs every 1 minute):

```
* * * * * /opt/jumpto-worker/scripts/cookies-refresh.sh >> /var/log/jumpto_cookie_refresh.log 2>&1
```

## 5. Deploy the worker

```sh
cp .env.example .env
# Edit .env to set BACKEND_URL and INTERNAL_API_KEY

docker compose up -d
```

## Verify

```sh
docker ps                          # Both chromium and jumpto-worker should be running
docker logs jumpto-worker --tail 20 # Check worker is consuming tasks
```

## 5. access the chromumuim from you machine
ssh -L 3001:127.0.0.1:3001 user@public_ip
