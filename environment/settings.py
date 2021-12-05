AREA_NAME = 'east_coast'
DAYS_CYCLE = 7
NUM_REQUESTS_PER_DAY = 40000
MIN_VOLUME = 1
MAX_VOLUME = 30


"reward settings"
TIME_REWARD = -0.5
FUEL_REWARD = -0.5
FUEL_PER_HOUR = 1.
REQUEST_REWARD = 0.0004
# PACKAGE_REWARD = 1.

"time settings"
SIMULATION_TIME = 48*60*60
NUM_WAIT_TIME = 1 #{0, 30, 60, 90, 120} min
WAIT_TIME_INTERVAL = 30*60

"vehicle and package settings"
NUM_TRUCKS = 20
NUM_HOPS = 10
RANK_MAX = 4
SILENT = True
TRUCK_SIZE = 30000
RANK_FILE = './inputs/RankList.pkl'
ETA_FILE = './inputs/eta_file'
KEY_FILE = './inputs/key_file'

"q-network settings"
NUM_OUTPUT = NUM_WAIT_TIME * NUM_HOPS + 1

