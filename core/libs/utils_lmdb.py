#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import io
import os

import lmdb
import numpy as np


class LMDBEngine:
    def __init__(self, lmdb_path, write=False):
        self._write = write
        self._manual_close = False
        self._lmdb_path = lmdb_path
        if write and not os.path.exists(lmdb_path):
            os.makedirs(lmdb_path)
        if write:
            self._lmdb_env = lmdb.open(lmdb_path, map_size=1099511627776)
            self._lmdb_txn = self._lmdb_env.begin(write=True)
        else:
            self._lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=True)
            self._lmdb_txn = self._lmdb_env.begin(write=False)
        # print('Load lmdb length:{}.'.format(len(self.keys())))

    def __getitem__(self, key_name):
        data = self.load(key_name)
        return data

    def __del__(self):
        if not self._manual_close:
            print("Writing engine not mannuly closed!")
            self.close()

    def load(self, key_name, **kwargs):
        payload = self._lmdb_txn.get(key_name.encode())
        if payload is None:
            raise KeyError("Key:{} Not Found!".format(key_name))
        numpy_buf = io.BytesIO(payload)
        numpy_data = np.load(numpy_buf, allow_pickle=True)
        return numpy_data

    def dump(self, key_name, payload):
        assert isinstance(payload, dict), "Payload must be a dict of numpy arrays."
        if not self._write:
            raise AssertionError("Engine Not Running in Write Mode.")
        if not hasattr(self, "_dump_counter"):
            self._dump_counter = 0
        if self.exists(key_name):
            print("Key:{} exsists!".format(key_name))
            return
        # dump numpy arrays
        numpy_buf = io.BytesIO()
        np.savez(numpy_buf, **payload)
        payload_encoded = numpy_buf.getvalue()
        self._lmdb_txn.put(key_name.encode(), payload_encoded)
        # counting
        self._dump_counter += 1
        if self._dump_counter % 2000 == 0:
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)

    def exists(self, key_name):
        if self._lmdb_txn.get(key_name.encode()):
            return True
        else:
            return False

    def delete(self, key_name):
        if not self._write:
            raise AssertionError("Engine Not Running in Write Mode.")
        if not self.exists(key_name):
            print("Key:{} Not Found!".format(key_name))
            return
        deleted = self._lmdb_txn.delete(key_name.encode())
        if not deleted:
            print("Delete Failed: {}!".format(key_name))
            return
        self._lmdb_txn.commit()
        self._lmdb_txn = self._lmdb_env.begin(write=True)

    def raw_load(self, key_name):
        raw_payload = self._lmdb_txn.get(key_name.encode())
        return raw_payload

    def raw_dump(self, key_name, raw_payload):
        if not self._write:
            raise AssertionError("Engine Not Running in Write Mode.")
        if not hasattr(self, "_dump_counter"):
            self._dump_counter = 0
        if self._lmdb_txn.get(key_name.encode()):
            print("Key:{} exsists!".format(key_name))
            return
        self._lmdb_txn.put(key_name.encode(), raw_payload)
        self._dump_counter += 1
        if self._dump_counter % 2000 == 0:
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)

    def keys(self):
        all_keys = list(self._lmdb_txn.cursor().iternext(values=False))
        all_keys = [key.decode() for key in all_keys]
        # print('Found data, length:{}.'.format(len(all_keys)))
        return all_keys

    def close(self):
        if self._write:
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)
        self._lmdb_env.close()
        self._manual_close = True
