import express from 'express';

const router = express.Router();

router.get('/hub-principles', (req, res) => {
  res.json({
    metadataLayer: 'postgres_including_column_comments',
    datasetRepresentation: 'metadata_layer',
    accessControl: 'roles_as_groups_app_maps_to_postgres_acl',
    credentialFlow: 'api_key_to_backend_to_short_lived_pg_credentials',
    computeTargets: ['local', 'spark', 'databricks'],
    apiKeyScopeOptions: ['user', 'project'],
    userApiKeyPrefix: 'w4h_sk_',
    userApiKeyHeaders: ['Authorization: Bearer <key>', 'Authorization: <jwt>', 'X-W4H-API-Key: <key>'],
    userApiKeyRotatePath: 'POST /users/me/api-key',
    userApiKeyRevokeSelfPath: 'DELETE /users/me/api-key',
    userApiKeyRevokeAdminPath: 'DELETE /users/user/:email/api-key',
    projectDbTokenHeader: 'X-W4H-Project-DB-Token',
    projectDbTokenMintPath: 'POST /projects/:id/db-token',
    projectDbTokenPurgeInactivePath: 'DELETE /projects/:id/db-tokens/purge-inactive',
    projectDbTokenListQuery: 'GET /projects/:id/db-tokens (active only); append ?history=1 for revoked/expired',
    notebookDataPaths: ['GET /datasets', 'GET /datasets/:id', 'GET /projects/:id']
  });
});

export default router;
