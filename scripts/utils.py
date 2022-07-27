from configparser import ConfigParser

class Config():
    def __init__(self, filename="config.ini"):
        self.config_file = filename
        self.read_config(self.config_file)


    def read_config(self, filename):
        """Reads data from the config file into a config class
        """
        config = ConfigParser()
        config.readfp(open(filename))

        # Read details mentioned in the config file
        # to the class object
        self.task_name = config.get('Meta', 'task_name')
        self.dataset_name = config.get('Meta','dataset_name')
        self.data_dir = config.get('Data','data_dir')
        self.data_file = config.get('Data','data_file')
        self.mode = config.get('Run','mode')
        self.model = config.get('Model','model')

if __name__ == "__main__":
    conf = Config()
    print(conf.task_name)
    print(conf.data_file)

