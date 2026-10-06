#!/bin/sh
set -e

echo "⏳ Waiting for MySQL port to be available..."
until nc -z "$DB_HOST" 3306 2>/dev/null; do
    echo "  MySQL port not ready yet, retrying in 2s..."
    sleep 2
done

echo "⏳ Giving MySQL 3 extra seconds to finish initialization..."
sleep 3

echo "✅ MySQL is ready!"
echo "🔧 Running admin account setup..."
python3 utils/create_admin.py

echo "🚀 Starting Flask app..."
exec python3 app.py
