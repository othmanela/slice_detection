import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T


class PatientEnvManager():
    def __init__(self, device, csv_file):
        self.device = device
        self.patients_data = pd.read_csv(csv_file)
        self.all_screens = None
        self.current_position = None
        self.patient = None
        self.l3_position = None
        self.z_size = None
        self.maxi = None
        self.mini = None
        self.done = False

    def reset(self, csv_row=None):
        self.all_screens = None
        self.current_position = None
        self.patient = None
        self.l3_position = None
        self.z_size = None
        self.maxi = None
        self.mini = None
        self.done = False
        if csv_row:
            sample = csv_row[1]
            self.patient = sample['name'].replace('.nii.gz', '')
            self.l3_position = sample['l3pos']
            self.z_size = sample['zsize']
            self.maxi = sample['maxi']
            self.mini = sample['mini']
        else:
            sample = self.patients_data.sample(n=1, replace=True)
            self.patient = sample['name'].values[0].replace('.nii.gz', '')
            self.l3_position = sample['l3pos'].values[0]
            self.z_size = sample['zsize'].values[0]
            self.maxi = sample['maxi'].values[0]
            self.mini = sample['mini'].values[0]
        self.current_position = np.random.randint(0, self.z_size)

    def close(self):
        self.all_screens = None
        self.current_position = None
        self.patient = None
        self.l3_position = None
        self.z_size = None
        self.maxi = None
        self.mini = None
        self.done = False

    def render(self):
        full_path = MIP_PATH + self.patient
        return np.load(full_path)

    @staticmethod
    def num_actions_available():
        return 2

    def take_action(self, action):
        if action.item() == 0:
            new_position = self.current_position + 1
        elif action.item() == 1:
            new_position = self.current_position - 1

        old_distance = abs(self.current_position - self.l3_position)
        new_distance = abs(new_position - self.l3_position)
        reward = np.sign(old_distance - new_distance)

        if new_position < 0 or new_position > self.z_size - 1:
            self.done = False
            reward = -1
            # don't update self.current_position
        elif new_position == self.l3_position:
            self.done = True
            self.current_position = new_position
            reward = 0.5
        else:
            self.done = False
            self.current_position = new_position
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.all_screens is None

    def get_state(self):
        if self.just_starting():
            scrn = self.get_processed_screen()
            fnl = scrn.detach().clone()
            return fnl
        else:
            s2 = self.get_processed_screen()
            final = s2.detach().clone()

            return final

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render()
        for i in range(screen.shape[1]):
            screen[self.current_position, i] = self.maxi

        if self.current_position < 100:
            new = np.full((200, 512), self.mini)
            start = 100 - self.current_position
            new[start:, :] = screen[:self.current_position + 100, :]
        elif self.current_position > (self.z_size - 100):
            new = np.full((200, 512), self.mini)
            start = self.current_position - 100
            end = self.z_size - self.current_position + 100
            new[0:end, :] = screen[start:, :]
        else:
            new = screen[self.current_position - 100: self.current_position + 100, :]
        tnsr = torch.tensor(new, dtype=torch.float)
        tnsr = tnsr.unsqueeze(0).unsqueeze(0).to(self.device)  # BCHW
        return tnsr

    def transform_screen_data(self, screen):
        resize = T.Compose([
            T.ToPILImage(),
            T.ToTensor()
        ])
        return resize(screen).unsqueeze(0).to(self.device)
