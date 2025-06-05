const sqlite3 = require('better-sqlite3')



const db = sqlite3('video_db.db', {
    
});

module.exports = {
    db
};