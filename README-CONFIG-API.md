# Database Configuration API

## User Roles

The clinical application has three roles:
- **admin** - System administrator with full access
- **researcher** - Clinical research staff with data access
- **clinician** - Clinical research staff with read only access

**⚠️ Database configuration endpoints are ADMIN ONLY (except during initial setup)**

## Database Initialization

When you configure the database for the first time, the system will automatically:
- Create the `users` table if it doesn't exist
- Create a default admin user (username: `admin`, password: `admin123`)
- Set up necessary indexes

**⚠️ Change the default admin password immediately after first login!**

## Endpoints

### 1. Check Configuration Status

**GET** `/config/status`

Check if the database is configured. **Public endpoint** (no authentication required).

**Response:**
```json
{
  "configured": true,
  "details": {
    "host": "localhost",
    "port": "5432",
    "database": "w4h_db",
    "user": "postgres"
  }
}
```

---

### 2. Configure Database

**POST** `/config/database` 🔒 **ADMIN ONLY** (public during initial setup)

Set database credentials securely. This will:
- Test the connection first
- Save credentials to `.env` file
- Update the connection pool immediately (no restart needed)
- Initialize database schema (create users table if needed)

**Headers:**
```
Authorization: <JWT token from login> (not required if DB not yet configured)
```

**Request Body:**
```json
{
  "host": "localhost",
  "port": "5432",
  "database": "w4h_db",
  "user": "postgres",
  "password": "your_password"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Database configuration saved and applied successfully.",
  "requiresRestart": false,
  "initialized": true
}
```

---

### 3. Test Connection

**GET** `/config/test` 🔒 **ADMIN ONLY**

Test the current database connection.

**Headers:**
```
Authorization: <JWT token from login>
```

**Response:**
```json
{
  "success": true,
  "message": "Database connection successful",
  "serverTime": "2026-02-08T10:30:00.000Z",
  "version": "PostgreSQL 14.5 on x86_64-apple-darwin21.6.0"
}
```

---

### 4. Reset Connection Pool

**POST** `/config/reset` 🔒 **ADMIN ONLY**

Manually reset the database connection pool (useful after manual `.env` changes).

**Headers:**
```
Authorization: <JWT token from login>
```

**Response:**
```json
{
  "success": true,
  "message": "Database pool reset successfully"
}
```

---

## Frontend Example

```javascript
// Check if database is configured (no auth required)
async function checkDbStatus() {
  const response = await fetch('http://localhost:2022/config/status');
  const data = await response.json();
  return data.configured;
}

// Configure database (no auth required during initial setup)
async function configureDatabase(credentials, token = null) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = token;
  }
  
  const response = await fetch('http://localhost:2022/config/database', {
    method: 'POST',
    headers,
    body: JSON.stringify(credentials)
  });
  return await response.json();
}

// Usage in Vue component
export default {
  data() {
    return {
      isConfigured: false,
      showConfigScreen: false
    }
  },
  async mounted() {
    this.isConfigured = await checkDbStatus();
    if (!this.isConfigured) {
      this.showConfigScreen = true;
    }
  }
}
```

---

## Security Notes

- Credentials are stored in `.env` file (ensure `.env` is in `.gitignore`)
- Only the status endpoint returns partial config details (no password)
- Test connection before saving to validate credentials
- Connection pool is automatically reset after configuration
- Database schema is automatically initialized with a default admin user
- **Change default admin password immediately after first login**
