const getUser = (username) => {
  const { FOUNDRY_CONFIG: FC } = require('../config');
  const fs = require('fs-extra');
  const users = fs.readJSONSync(FC.users_file);
  const found = users.find(d => d.username === username);
  
  return {
    username: found ? found.username : '',
    name: found ? found.name : '',
    email: found ? found.email : ''
  };
}

exports.getUser = getUser;