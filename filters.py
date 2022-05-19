import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def _gaussian_kernel2d(sigma, kernel_size):
    coordinates = torch.arange(kernel_size).float()
    x_coordinates = coordinates.repeat(kernel_size)\
            .view(kernel_size, kernel_size).float()
    y_coordinates = x_coordinates.t()
    mean = (kernel_size - 1) / 2.
    var = sigma ** 2.
    xy_coordinates = torch.stack([x_coordinates, y_coordinates], dim=-1) - mean
    kernel = torch.exp(-0.5 / var * torch.sum(xy_coordinates ** 2, dim=-1))
    return kernel / torch.sum(kernel)


def center_surround(sigma_1, sigma_2, k, kernel_size=41):
    center = _gaussian_kernel2d(sigma_1, kernel_size)
    surround = _gaussian_kernel2d(sigma_2, kernel_size)
    dog = (center - k * surround).to(device).float()
    return dog


def oriented_filters(filter_length, number_of_filters, pad_width, center_shift=0):
    horizontal_filter_coordinates = np.vstack([
        center_shift * np.ones(filter_length),
        np.arange(-(filter_length - 1) / 2, (filter_length - 1) / 2 + 1, 1)])
    filter_bank_list = []
    for theta in np.linspace(0, np.pi, number_of_filters, endpoint=False):
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]])
        orientation_filter = np.zeros((filter_length, filter_length))
        coordinates = (rot_matrix.dot(horizontal_filter_coordinates) +
                       (filter_length - 1) / 2).astype(int)
        orientation_filter[coordinates[0], coordinates[1]] = 1.
        orientation_filter /= np.sum(orientation_filter)
        padded_filter = np.pad(orientation_filter, pad_width)
        filter_bank_list.append(padded_filter)
    return torch.tensor(filter_bank_list).to(device).float()

def v1_filters(filter_length, number_of_filters, pad_width, filter_type='single'):
    if filter_type == 'double':
        filters_1 = oriented_filters(
            filter_length=filter_length,
            number_of_filters=number_of_filters,
            pad_width=pad_width,
            center_shift=0.5)[:, None, :, :]
        filters_2 = oriented_filters(
            filter_length=filter_length,
            number_of_filters=number_of_filters,
            pad_width=pad_width,
            center_shift=-0.5)[:, None, :, :]
    elif filter_type == 'single':
        filters_1 = oriented_filters(
            filter_length=filter_length,
            number_of_filters=number_of_filters,
            pad_width=pad_width,
            center_shift=0.0)[:, None, :, :]
        filters_2 = 0.0 * filters_1
    else:
        raise ValueError(f'filter_type: {filter_type} is not recognized.')
    
    return torch.cat([
        torch.cat([filters_1, filters_2], dim=1),
        torch.cat([filters_2, filters_1], dim=1)],
        dim=0)
