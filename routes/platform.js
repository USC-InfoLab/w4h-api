import express from 'express';
import { authenticate } from '../auth.js';

const router = express.Router();

/**
 * GET /platform/hub-principles
 * Non-secret alignment notes for clients (metadata, ACL, credentials).
 * Datasets (catalog projects) remain the source of truth; this is descriptive only.
 */
router.get('/hub-principles', authenticate, (req, res) => {
  res.json({
    metadataLayer: 'postgres_including_column_comments',
    datasetRepresentation: 'metadata_layer',
    accessControl: 'roles_as_groups_app_maps_to_postgres_acl',
    credentialFlow: 'api_key_to_backend_to_short_lived_pg_credentials',
    computeTargets: ['local', 'spark', 'databricks'],
    apiKeyScopeOptions: ['user', 'project']
  });
});

export default router;
