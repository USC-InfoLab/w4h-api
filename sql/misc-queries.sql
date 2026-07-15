SELECT * FROM w4h.users
ORDER BY id ASC

SELECT * FROM w4h.datasets
ORDER BY id ASC 

SELECT * FROM w4h.docs
ORDER BY id ASC

-- Remove platform admins from project_members (they belong in projects.created_by, not the team list)
-- DELETE FROM w4h.project_members pm
-- USING w4h.users u
-- WHERE pm.user_id = u.id AND u.role = 'admin';

SELECT * FROM w4h.studies
ORDER BY id ASC