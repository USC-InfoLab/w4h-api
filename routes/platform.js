import express from 'express';
import { authenticate } from '../auth.js';

const router = express.Router();

router.get('/hub-principles', authenticate, (req, res) => {
  res.json({
    metadataLayer: 'postgres_including_column_comments',
    datasetRepresentation: 'metadata_layer',
    accessControl: 'roles_as_groups_app_maps_to_postgres_acl',
    credentialFlow: 'api_key_to_backend_to_short_lived_pg_credentials',
    computeTargets: ['local', 'spark', 'databricks'],
    apiKeyScopeOptions: ['user', 'project'],
    notebookAuth: {
      userApiKeyHeaders: [
        'x-w4h-api-key: <key>',
        'Authorization: W4H-Api-Key <key>'
      ],
      jwtAuthorization: 'Authorization: <token returned by POST /users/login>',
      projectDbTokenHeader: 'x-w4h-project-db-token',
      mintUserApiKey: 'POST /users/me/api-key (requires JWT)',
      revokeUserApiKeySelf: 'DELETE /users/me/api-key (requires JWT)',
      revokeUserApiKeyAdmin: 'DELETE /users/user/:email/api-keys (requires admin JWT)',
      mintProjectDbToken: 'POST /projects/:projectId/db-token (requires JWT)',
      routesWithUserApiKey: [
        'GET /projects',
        'GET /projects/:id',
        'GET /datasets',
        'GET /datasets/:id',
        'GET /datasets/projects',
        'GET /datasets/tableinfo'
      ]
    }
  });
});

export default router;
