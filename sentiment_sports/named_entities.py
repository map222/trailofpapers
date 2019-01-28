NICKNAMES_OLDTIMERS = set('lma kd lbj klove drose kobe mj magic '\
                          'bird wilt chamberlain tt shump kat melo steph '\
                          'k-love d-wade dwade boogie bosh iggy mcbuckets '\
                          'cp3 shaq dray hakeem kareem zo lamar odom delly chuck barkley'.lower().split())
TEAMS = set('Atlanta Hawks Boston Celtics Brooklyn Nets Charlotte Bobcats ' \
                  'Chicago Bulls Cleveland Cavaliers cavs Dallas Mavericks mavs Denver Nuggets ' \
                  'Detroit Pistons Golden State Warriors Houston Rockets Indiana '\
                  'Pacers Los Angeles LA Clippers LA Lakers LAL Memphis Grizzlies Miami Heat '\
                  'Milwaukee Bucks Minnesota Timberwolves  New Orleans Hornets New York Knicks NYK '\
                  'Oklahoma City Thunder Orlando Magic Philadelphia Sixers 76ers Phoenix '\
                  'Suns Portland Trail Blazers Sacramento Kings '\
                  'San Antonio Spurs Toronto Raptors Utah Jazz Washington Wizards wiz '\
                  'NBA asg'.lower().split() )
NON_PLAYERS = TEAMS.union(set('KOC woj shams ainge'.lower().split()) )
UPPER_ENTITIES = {'Black', 'Ball' 'Buddy', 'Grant', 'House', 'Smart', 'Holiday', 'Love', 'Rose',
                  'Smart', 'Stone', 'Temple', 'Wall', 'Will', 'Wear', 'Case', 'New', 'Little',
                  'Shorts', 'City', 'Will', 'Young', 'Gay', 'Price', 'World', 'Peace', 'Long', 'Semi',
                  'Brown', 'Green', 'Blue', 'White', 'Brand', 'Early'}
